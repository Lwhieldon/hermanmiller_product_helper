from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List
import re
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate

INDEX_NAME = "hermanmiller-product-helper"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """You are a helpful product assistant for Herman Miller pricing catalogs.
Always respond clearly and concisely.
If the user asks about a pricing breakdown or part number (e.g. FT123), include any related pricing tables and product images.

{context}

Question: {question}
Helpful Answer:"""
    )

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(search_kwargs={"k": 5}),
        prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke({
        "input": query,
        "chat_history": chat_history,
        "question": query
    })

    return result


def fetch_chunks_by_part_number(part_numbers: List[str], k: int = 100):
    docsearch = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large")
    )
    all_chunks = docsearch.similarity_search("product", k=k)

    filtered = []
    for doc in all_chunks:
        meta = doc.metadata
        if any(pn in meta.get("part_numbers", []) for pn in part_numbers):
            filtered.append(doc)
    return filtered


def process_response(llm_response: dict) -> dict:
    sources = []
    images = []
    fallback_chunks = []
    raw_answer = llm_response.get("answer", "").strip()
    question = llm_response.get("question", "")

    # Extract FT codes from question
    query_parts = re.findall(r"\bFT\d{3,4}\b", question)

    # 1. Collect LLM-returned chunks
    for doc in llm_response.get("source_documents", []):
        if hasattr(doc, "metadata"):
            meta = doc.metadata
            page = meta.get("page", "Unknown")

            if "image_path" in meta:
                caption = meta.get("heading") or meta.get("prev_heading") or f"Page {page}"
                images.append({"path": meta["image_path"], "caption": caption})

            if any(pn in meta.get("part_numbers", []) for pn in query_parts):
                fallback_chunks.append(doc)

            sources.append({
                "page": page,
                "source": meta.get("source", "Unknown"),
                "heading": meta.get("heading"),
                "prev_heading": meta.get("prev_heading"),
            })

    # 2. Fetch matching chunks from Pinecone by metadata
    if query_parts:
        matched_chunks = fetch_chunks_by_part_number(query_parts, k=200)
        for doc in matched_chunks:
            meta = doc.metadata
            if doc not in fallback_chunks:
                fallback_chunks.append(doc)
            if "image_path" in meta:
                caption = meta.get("heading") or meta.get("prev_heading") or f"Page {meta.get('page', 'Unknown')}"
                image_data = {"path": meta["image_path"], "caption": caption}
                if image_data not in images:
                    images.append(image_data)
            if meta.get("source") or meta.get("page"):
                sources.append({
                    "page": meta.get("page", "Unknown"),
                    "source": meta.get("source", "Unknown"),
                    "heading": meta.get("heading"),
                    "prev_heading": meta.get("prev_heading"),
                })

    # 3. If LLM didn't return a confident answer but we found context
    if not raw_answer or "not listed" in raw_answer.lower() or "no information" in raw_answer.lower():
        if fallback_chunks or images:
            raw_answer = (
                "Here is what I found for the product(s) you asked about. See the illustrations and pricing context below."
            )

    return {
        "answer": format_answer_as_markdown_table(raw_answer),
        "sources": sources,
        "images": images[:4]  # Limit carousel to 4 images
    }


def format_answer_as_markdown_table(text: str) -> str:
    lines = text.strip().splitlines()
    table_lines = [line for line in lines if "$" in line and "|" in line]

    if not table_lines:
        return text

    header = table_lines[0]
    separator = "|".join("---" if part.strip() else "" for part in header.split("|"))
    return "\n".join([header, separator] + table_lines[1:])


if __name__ == "__main__":
    query = "Show me FT123 and any available illustrations or pricing."
    response = process_response(run_llm(query))
    print(response)
