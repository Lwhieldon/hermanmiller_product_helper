import re
import logging
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

logging.basicConfig(level=logging.INFO)

prompt = ChatPromptTemplate.from_template("""
You are a helpful product expert for Herman Miller. Use the chat history and provided context to answer the latest question.

- If a product or part number appears with pricing, output it as a clean markdown table.
- Include any variations in finishes (e.g., Metallic Paint), dimensions, and options.
- Only include prices and specs that are explicitly found in the context.
- If images are available for a product or part number, include them in the response with captions and page references.
- Do not invent or guess missing values — leave them blank or say "not found".
- If prices or product specs are partially available, try building a markdown table with as much as possible.
- If something is unclear, note it as "unknown" or "not listed" rather than rejecting the response.
- Support answers for features like MicrobeCare™, surface edge options, bracket types, and finishes even if not linked to a part.

Examples:
Q: Breakdown in a pricing table FT123. In your response, output the pricing in table format.
A: [Table based on actual PDF pricing]

Q: What are the MicrobeCare™ options and what does this coating protect against?
A: MicrobeCare™ is an antimicrobial coating that protects against mold, mildew, and bacteria. Available on panels and screen surfaces.

Q: Do you have any images related to FT292?
A: Yes, see below. These are product illustrations from the price book.

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

def log_part_result(part_numbers: List[str], found_docs: List[Document]):
    found_parts = set()
    for d in found_docs:
        for p in d.metadata.get("part_numbers", []):
            found_parts.add(p.lower())
    for pn in part_numbers:
        if pn.lower() in found_parts:
            logging.info(f"✅ Found part {pn}")
        else:
            logging.warning(f"❌ DID NOT find part {pn}")

def classify_query(query: str) -> str:
    query = query.lower()
    if any(term in query for term in ["price", "pricing", "cost", "$"]):
        return "pricing"
    elif any(term in query for term in ["image", "illustration", "diagram", "picture", "photo", "drawing"]):
        return "image"
    elif any(term in query for term in ["material", "finish", "surface", "edge", "veneer", "glass", "fabric", "coating", "bracket", "microbecare", "attachment"]):
        return "feature"
    return "general"

def rank_docs(docs: List[Document]) -> List[Document]:
    def score(doc):
        meta = doc.metadata
        score = 0
        if meta.get("is_pricing_table"): score += 3
        if meta.get("is_feature_block"): score += 2
        if "part_numbers" in meta: score += 1
        return score
    return sorted(docs, key=score, reverse=True)

def run_llm(query: str, chat_history: List[str] = []) -> Dict[str, Any]:
    classification = classify_query(query)
    part_numbers = extract_part_numbers_from_query(query)

    docs = []
    try:
        # Step 1: Try filtered vector search
        if part_numbers:
            docs = retriever.vectorstore.similarity_search(
                query=query,
                k=30,
                filter={"part_numbers": {"$in": [pn.lower() for pn in part_numbers]}}
            )
            log_part_result(part_numbers, docs)

        elif classification == "feature":
            docs = retriever.vectorstore.similarity_search(
                query=query,
                k=20,
                filter={"is_feature_block": True}
            )

        # Step 2: If docs are sparse, fallback to unfiltered vector search
        if not docs or not any(len(d.page_content.strip()) > 40 for d in docs):
            docs = retriever.vectorstore.similarity_search(query, k=30)

    except Exception as e:
        logging.error(f"❌ Filtered search failed: {e}")
        docs = retriever.vectorstore.similarity_search(query, k=30)

    docs = rank_docs(docs)

    # Search for image docs separately
    image_docs = []
    if part_numbers:
        try:
            image_docs = retriever.vectorstore.similarity_search(
                query="illustration",
                k=40,
                filter={
                    "part_numbers": {"$in": [pn.lower() for pn in part_numbers]},
                    "image_path": {"$exists": True}
                }
            )
        except Exception as e:
            logging.warning(f"⚠️ Image search failed: {e}")

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
            match = any(pn.lower() in doc_parts for pn in part_numbers)
            fuzzy_match = any(any(pn.lower() in part for part in doc_parts) for pn in part_numbers)
            if not match and not fuzzy_match:
                continue
        images.append({
            "path": meta["image_path"],
            "caption": meta.get("caption", "Product illustration"),
            "page": meta.get("page"),
            "source": meta.get("source"),
            "part_numbers": meta.get("part_numbers", [])
        })
    return images
