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
    elif any(term in query for term in ["image", "illustration", "diagram", "picture", "photo", "drawing"]):
        return "image"
    elif any(term in query for term in ["material", "finish", "surface", "edge", "veneer", "glass", "fabric", "coating", "bracket", "microbecare", "attachment"]):
        return "feature"
    return "general"

prompt = ChatPromptTemplate.from_template("""
You are a helpful product expert for Herman Miller. Use the chat history and provided context to answer the latest question.

- If a product or part number appears with pricing, output it as a clean markdown table.
- If the user asks about finishes, materials, MicrobeCare™, or bracket options, include them from any feature blocks or descriptive text.
- If images are available, include them with captions and page numbers.
- If something is not found in the context, say "not listed" or "not available".
- Never make up prices or specifications.

Examples:
Q: Do you have any images related to FT292?
A: Yes, image illustrations are available — see below.

Q: What are the MicrobeCare™ options and what does it protect against?
A: MicrobeCare™ is an antimicrobial coating. Available for surfaces: yes. Protects against: mold, mildew, bacteria.

Q: What are the door material options for sliding door storage units?
A: Veneer, glass, painted steel, thermoplastic laminate.

CHAT HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{input}

Helpful Answer:
""")

MAX_TOKENS = 100000

def truncate_docs(docs: List[Document], max_tokens: int = MAX_TOKENS) -> str:
    total_tokens = 0
    context_parts = []
    for doc in docs:
        text = doc.page_content
        token_count = len(text.split())
        if total_tokens + token_count > max_tokens:
            break
        context_parts.append(text)
        total_tokens += token_count
    return "\n\n".join(context_parts)

def extract_part_numbers_from_query(query: str) -> List[str]:
    query = query.upper()
    parts = re.split(r"[,+&/]+|\band\b", query)
    return list({pn.strip() for pn in parts if re.match(r"\b[A-Z]{2}\d{3,4}\b", pn.strip())})

def format_chat_history(history: List[str]) -> str:
    formatted = []
    for role, text in history[-10:]:
        prefix = "User" if role == "human" else "Assistant"
        formatted.append(f"{prefix}: {text}")
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

    image_docs = []
    if part_numbers:
        try:
            image_docs = retriever.vectorstore.similarity_search(
                query="illustration",
                k=50,
                filter={"part_numbers": {"$in": [pn.lower() for pn in part_numbers]},
                        "image_path": {"$exists": True}}
            )
        except Exception as e:
            print("⚠️ Image metadata fallback failed:", e)

    seen_ids = set()
    all_docs = []
    for doc in docs + image_docs:
        doc_id = (
            doc.metadata.get("page"),
            doc.metadata.get("image_path"),
            doc.metadata.get("chunk_index")
        )
        if doc_id not in seen_ids:
            all_docs.append(doc)
            seen_ids.add(doc_id)

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
    return [doc.metadata for doc in docs if doc.metadata.get("page")]

def get_relevant_images_from_docs(docs: List[Document], part_numbers: List[str] = None) -> List[Dict[str, Any]]:
    images = []
    for doc in docs:
        meta = doc.metadata
        if "image_path" not in meta:
            continue
        if part_numbers:
            doc_parts = meta.get("part_numbers", [])
            if not any(pn.lower() in doc_parts for pn in part_numbers):
                continue
        images.append({
            "path": meta["image_path"],
            "caption": meta.get("caption", "Product illustration"),
            "page": meta.get("page"),
            "source": meta.get("source"),
            "part_numbers": meta.get("part_numbers", [])
        })
    return images
