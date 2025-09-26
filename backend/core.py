import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME", "hermanmiller-product-helper")

retriever = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    text_key="page_content"
).as_retriever()

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

def classify_query(query: str) -> str:
    query = query.lower()
    if any(term in query for term in ["price", "pricing", "cost", "$"]):
        return "pricing"
    elif any(term in query for term in ["image", "illustration", "diagram", "picture"]):
        return "image"
    elif any(term in query for term in [
        "material", "finish", "surface", "edge", "microbecare", "bracket",
        "veneer", "glass", "fabric", "top cap"
    ]):
        return "feature"
    else:
        return "general"

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful product expert for Herman Miller. Use the chat history and provided context to answer the latest question accurately and helpfully.

    **IMPORTANT GUIDELINES:**
    - If a product or part number appears with pricing, output it as a clean markdown table.
    - Include any variations in finishes (e.g., Metallic Paint), dimensions, and options found in the context.
    - Only include prices and specs that are explicitly found in the context - do not guess or extrapolate.
    - If images are available for a product or part number, mention them in your response.
    - Do not invent or guess missing values — be honest about what information is not available.
    - If prices or product specs are partially available, build a markdown table with available information and clearly mark missing data as "not specified".
    - If something is unclear or contradictory, acknowledge this rather than making assumptions.
    - Always provide helpful, actionable information when possible.
    - If the context doesn't contain relevant information, politely explain this and suggest what information would be helpful.

    **RESPONSE FORMAT:**
    - Use clear, professional language
    - Structure information logically (specifications, pricing, features, etc.)
    - Use markdown tables for pricing and technical specifications
    - Provide specific part numbers and page references when available

    CHAT HISTORY:
    {chat_history}

    CONTEXT:
    {context}

    QUESTION:
    {input}

    Helpful Answer:
    """
)

MAX_TOKENS = 100000

def truncate_docs(docs: List[Document], max_tokens: int = MAX_TOKENS) -> str:
    """
    Intelligently truncate documents to fit within token limit.
    Prioritizes meaningful content and maintains document boundaries.
    """
    if not docs:
        return ""
    
    total_tokens = 0
    context_parts = []
    
    for doc in docs:
        text = doc.page_content.strip()
        if not text:  # Skip empty documents
            continue
            
        token_count = len(text.split())  # Approximate token count
        
        # If this single document would exceed the limit, truncate it
        if token_count > max_tokens:
            words = text.split()
            truncated_words = words[:max_tokens]
            text = " ".join(truncated_words)
            context_parts.append(text)
            break  # This document filled our entire budget
            
        # If adding this document would exceed the limit, stop here
        if total_tokens + token_count > max_tokens:
            break
            
        context_parts.append(text)
        total_tokens += token_count
    
    result = "\n\n".join(context_parts)
    
    # Ensure we return something meaningful
    if not result.strip():
        # Try to get at least some content from the first non-empty document
        for doc in docs:
            text = doc.page_content.strip()
            if text:
                words = text.split()
                # Take at least 50 words if available
                min_words = min(50, len(words))
                result = " ".join(words[:min_words])
                break
    
    return result

def extract_part_numbers_from_query(query: str) -> List[str]:
    return re.findall(r"\b[A-Z]{2}\d{3,4}\b", query.upper())

def format_chat_history(history: List[str]) -> str:
    """
    Format chat history for LLM context.
    Handles various input formats and ensures consistent output.
    """
    if not history:
        return ""
    
    formatted = []
    # Take last 10 exchanges for context (20 total messages max)
    recent_history = history[-20:] if len(history) > 20 else history
    
    for item in recent_history:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                role, text = item[0], item[1]
                if role == "human":
                    formatted.append(f"User: {text}")
                elif role in ["ai", "assistant"]:
                    formatted.append(f"Assistant: {text}")
                else:
                    # Handle unknown roles gracefully
                    formatted.append(f"{role.title()}: {text}")
            else:
                # Handle malformed history items
                print(f"⚠️ Malformed chat history item: {item}")
                continue
        except Exception as e:
            print(f"⚠️ Error processing chat history item {item}: {e}")
            continue
    
    return "\n".join(formatted)

def run_llm(query: str, chat_history: List[str] = []) -> Dict[str, Any]:
    classification = classify_query(query)
    part_numbers = extract_part_numbers_from_query(query)

    docs = []
    retrieval_method = "none"
    
    try:
        if part_numbers:
            docs = retriever.vectorstore.similarity_search(
                query=query,
                k=10,
                filter={"part_numbers": {"$in": [pn.lower() for pn in part_numbers]}}
            )
            retrieval_method = f"part_numbers({','.join(part_numbers)})"
            print(f"✅ Retrieved {len(docs)} docs for part numbers: {part_numbers}")
            
            # Check for meaningful content (more than just whitespace/very short content)
            meaningful_docs = [d for d in docs if len(d.page_content.strip()) > 20]
            if not meaningful_docs:
                print("⚠️ Part number docs too short — retrying with semantic search")
                docs = retriever.vectorstore.similarity_search(query, k=10)
                retrieval_method = f"semantic_fallback_from_part_numbers"
                
        elif classification == "feature":
            docs = retriever.vectorstore.similarity_search(
                query=query,
                k=10,
                filter={"is_feature_block": True}
            )
            retrieval_method = "feature_blocks"
            print(f"✅ Retrieved {len(docs)} feature docs with feature_block filter")
            
            # If no feature blocks found, fall back to semantic search
            if not docs:
                print("⚠️ No feature blocks found — using semantic search")
                docs = retriever.vectorstore.similarity_search(query, k=10)
                retrieval_method = "semantic_fallback_from_features"
                
    except Exception as e:
        print(f"⚠️ Filtered search failed: {e}")
        docs = []

    # Final fallback if we still have no docs
    if not docs:
        print("⚠️ No docs from filtered search — falling back to default retrieval")
        docs = retriever.invoke(query)
        retrieval_method = "default_retriever"

    # Filter out completely empty documents
    docs = [doc for doc in docs if doc.page_content.strip()]
    
    if not docs:
        print("❌ No meaningful documents found after filtering")
        return {
            "answer": "I don't have enough information to answer your question. Please try rephrasing or asking about specific Herman Miller products.",
            "type": classification,
            "sources": [],
            "images": []
        }

    context = truncate_docs(docs)
    chat_history_text = format_chat_history(chat_history)
    
    print(f"📄 Using {len(docs)} docs via {retrieval_method}, context: {len(context)} chars")

    # Ensure we have meaningful context
    if not context.strip():
        print("❌ Empty context after truncation")
        return {
            "answer": "I found some documents but couldn't extract meaningful content. Please try rephrasing your question.",
            "type": classification,
            "sources": get_relevant_sources_from_docs(docs),
            "images": []
        }

    try:
        response = (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "input": query,
            "chat_history": chat_history_text
        })
        
        # Validate response
        if not response or not response.strip():
            print("❌ LLM returned empty response")
            response = "I found relevant information but couldn't generate a proper response. Please try rephrasing your question."
            
    except Exception as e:
        print(f"❌ LLM invocation failed: {e}")
        response = "I encountered an error while processing your request. Please try again or rephrase your question."

    return {
        "answer": response,
        "type": classification,
        "sources": get_relevant_sources_from_docs(docs),
        "images": get_relevant_images_from_docs(docs) if classification in ["image", "pricing"] else []
    }

def get_relevant_sources_from_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    return [ {
        "page": doc.metadata.get("page"),
        "pages": doc.metadata.get("pages"),
        "heading": doc.metadata.get("heading"),
        "prev_heading": doc.metadata.get("prev_heading")
    } for doc in docs ]

def get_relevant_images_from_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    return [ {
        "path": doc.metadata["image_path"],
        "caption": doc.metadata.get("caption"),
        "page": doc.metadata.get("page")
    } for doc in docs if "image_path" in doc.metadata ]
