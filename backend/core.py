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

INDEX_NAME_2 = os.getenv("INDEX_NAME_2", "hermanmiller-product-helper-images")

retriever = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME_2,
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
    You are a helpful product expert for Herman Miller. Use the chat history and provided context to answer the latest question.

    - If a product or part number appears with pricing, output it as a clean markdown table.
    - Include any variations in finishes (e.g., Metallic Paint), dimensions, and options.
    - Only include prices and specs that are explicitly found in the context.
    - If images are available for a product or part number, include them in the response with captions.
    - Do not invent or guess missing values — leave them blank or say "not found".
    - If prices or product specs are partially available, try building a markdown table with as much as possible.
    - If something is unclear, note it as "unknown" or "not listed" rather than rejecting the response.

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
    total_tokens = 0
    context_parts = []
    for doc in docs:
        text = doc.page_content
        token_count = len(text.split())  # Approximate
        if total_tokens + token_count > max_tokens:
            break
        context_parts.append(text)
        total_tokens += token_count
    return "\n\n".join(context_parts)

def extract_part_numbers_from_query(query: str) -> List[str]:
    return re.findall(r"\b[A-Z]{2}\d{3,4}\b", query.upper())

def format_chat_history(history: List[str]) -> str:
    formatted = []
    for role, text in history[-10:]:  # Last 10 exchanges for context
        if role == "human":
            formatted.append(f"User: {text}")
        else:
            formatted.append(f"Assistant: {text}")
    return "\n".join(formatted)

def run_llm(query: str, chat_history: List[str] = []) -> Dict[str, Any]:
    classification = classify_query(query)
    part_numbers = extract_part_numbers_from_query(query)

    docs = []
    try:
        if part_numbers:
            docs = retriever.vectorstore.similarity_search(
                query=query,
                k=20,
                filter={"part_numbers": {"$in": [pn.lower() for pn in part_numbers]}}
            )
            if not any(len(d.page_content.strip()) > 30 for d in docs):
                print("⚠️ Docs too short — retrying without filter")
                docs = retriever.vectorstore.similarity_search(query, k=20)
        elif classification == "feature":
            docs = retriever.vectorstore.similarity_search(
                query=query,
                k=20,
                filter={"is_feature_block": True}
            )
        else:
            docs = retriever.vectorstore.similarity_search(query, k=20)
    except Exception as e:
        print("⚠️ Filtered search failed:", e)

    # Fallback image retrieval if needed
    image_docs = []
    if part_numbers:
        try:
            image_docs = retriever.vectorstore.similarity_search(
                query=" ".join(part_numbers),
                k=10,
                filter={"image_path": {"$exists": True}}
            )
            image_docs = [
                doc for doc in image_docs
                if any(pn.lower() in doc.metadata.get("part_numbers", []) for pn in part_numbers)
            ]
        except Exception as e:
            print("⚠️ Image fallback retrieval failed:", e)

    # Deduplicate and merge
    all_docs = docs + [d for d in image_docs if d not in docs]

    context = truncate_docs(all_docs)
    chat_history_text = format_chat_history(chat_history)

    response = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context,
        "input": query,
        "chat_history": chat_history_text
    })

    return {
        "answer": response,
        "type": classification,
        "sources": get_relevant_sources_from_docs(all_docs),
        "images": get_relevant_images_from_docs(all_docs, part_numbers)
    }

def get_relevant_sources_from_docs(docs: List[Document]) -> List[Dict[str, Any]]:
    return [{
        "page": doc.metadata.get("page"),
        "pages": doc.metadata.get("pages"),
        "heading": doc.metadata.get("heading"),
        "prev_heading": doc.metadata.get("prev_heading")
    } for doc in docs]

def get_relevant_images_from_docs(docs: List[Document], part_numbers: List[str] = None) -> List[Dict[str, Any]]:
    images = []
    for doc in docs:
        if "image_path" not in doc.metadata:
            continue
        if part_numbers:
            doc_parts = doc.metadata.get("part_numbers", [])
            if not any(pn.lower() in doc_parts for pn in part_numbers):
                continue
        images.append({
            "path": doc.metadata["image_path"],
            "caption": doc.metadata.get("caption", "Product illustration"),
            "page": doc.metadata.get("page")
        })
    return images
