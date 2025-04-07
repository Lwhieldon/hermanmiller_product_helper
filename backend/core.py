# core.py

import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
INDEX_NAME = os.getenv("INDEX_NAME", "hermanmiller-product-helper")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that answers questions about Herman Miller products using specifications, images, and pricing data.
When possible, you provide a pricing table in markdown format when asked about pricing. But only return pricing tables if it will helps the user.
{context}

Question: {question}
Helpful Answer:
""",
)

def qa_chain(docs: List[Document], question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = custom_prompt.format(context=context, question=question)
    return llm.invoke(prompt)

def run_llm(query: str) -> Dict[str, Any]:
    docs = retriever.invoke(query)
    answer = qa_chain(docs, query)
    return {
        "question": query,
        "source_documents": docs,
        "llm_answer": answer,
    }

def process_response(response: Dict[str, Any]) -> Dict[str, Any]:
    raw_answer = str(response.get("llm_answer", "")).strip()
    docs = response.get("source_documents", [])
    question = response.get("question", "").lower()

    part_numbers = re.findall(r"\bFT\d{3,4}\b", question)
    images = []
    sources = []

    def extract_image(doc):
        meta = doc.metadata
        return {
            "path": meta["image_path"],
            "caption": meta.get("description") or meta.get("heading") or meta.get("prev_heading") or f"Page {meta.get('page')}"
        }

    for doc in docs:
        meta = doc.metadata
        if meta.get("image_path"):
            img_data = extract_image(doc)
            if img_data not in images:
                images.append(img_data)

        sources.append({
            "page": meta.get("page", "Unknown"),
            "source": meta.get("source", "Unknown"),
            "heading": meta.get("heading"),
            "prev_heading": meta.get("prev_heading")
        })

    return {
        "answer": format_answer_as_markdown_table(raw_answer),
        "sources": sources,
        "images": images[:4]  # Limit to 4 images
    }

def format_answer_as_markdown_table(text: str) -> str:
    lines = text.strip().splitlines()
    table_lines = [line for line in lines if re.search(r"\|.*\d", line)]
    if not table_lines:
        return text
    header = table_lines[0]
    separator = "|".join(["---"] * len(header.split("|")))
    return "\n".join([header, separator] + table_lines[1:])

if __name__ == "__main__":
    query = "Tell me what products you can provide an image for?"
    response = process_response(run_llm(query))
    print(response)
