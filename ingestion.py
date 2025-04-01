from dotenv import load_dotenv
import nltk
import os
import tempfile
import requests
import re
from itertools import groupby

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    FireCrawlLoader, 
    PyMuPDFLoader, 
    UnstructuredPDFLoader
)

# Download necessary NLTK models
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Load environment variables
load_dotenv()

# Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Download the PDF
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, os.path.basename(url))
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    return None

# Flatten nested metadata dicts
def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, (str, int, float, bool)):
            items[new_key] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            items[new_key] = v
        else:
            items[new_key] = str(v)
    return items

# Clean metadata for indexing
def clean_metadata(metadata):
    return flatten_dict(metadata)

# Merge wrapped rows in tables for better semantic coherence
def merge_wrapped_rows(text):
    lines = text.split("\n")
    merged_lines = []
    for line in lines:
        if merged_lines and line.startswith(" "):
            merged_lines[-1] += " " + line.strip()
        else:
            merged_lines.append(line)
    return "\n".join(merged_lines)

# Expanded heading pattern matcher
SECTION_PATTERNS = [
    r"Walls", r"Work Surfaces", r"Storage", r"Screens", r"Lighting",
    r"Installation", r"Warranty", r"Connectors", r"Terms and Conditions"
]

def extract_section_heading(text):
    for pattern in SECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return "General"

# Extract part/product code like FT110.
def extract_product_code(text):
    match = re.search(r"(FT\d{3}\.)", text)
    return match.group(1) if match else None

# Identify if chunk contains price list blocks
def is_price_block(text):
    return re.search(r"FT\d{3}\.", text) and re.search(r"\$\d{2,}", text)

# Group documents by pricing table structure
def group_table_related_docs(docs):
    grouped_docs = []
    group = []
    for doc in docs:
        if is_price_block(doc.page_content):
            group.append(doc)
        else:
            if group:
                text = "\n".join([d.page_content for d in group])
                grouped_docs.append(Document(page_content=text, metadata=group[0].metadata))
                group = []
            grouped_docs.append(doc)
    if group:
        text = "\n".join([d.page_content for d in group])
        grouped_docs.append(Document(page_content=text, metadata=group[0].metadata))
    return grouped_docs

# Ingestion pipeline
def ingest_docs():
    INDEX_NAME = os.getenv("INDEX_NAME")
    documents_base_urls = [
        "https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf"
    ]

    for url in documents_base_urls:
        print(f"Loading {url}")

        if url.endswith(".pdf"):
            pdf_path = download_pdf(url)
            if pdf_path:
                unstructured_loader = UnstructuredPDFLoader(
                    file_path=pdf_path,
                    mode="elements",
                    strategy="hi_res",
                    unstructured_kwargs={
                        "detect_tables": True,
                        "use_image": True,
                        "image_quality": "high",
                        "include_metadata": True,
                        "image_processing": "auto",
                        "include_text": True,
                        "text_quality": "high",
                        "include_page_breaks": True,
                        "page_break_strategy": "auto",
                        "ocr_languages": ["eng"]
                    }
                )
                pymupdf_loader = PyMuPDFLoader(pdf_path)

                raw_documents = unstructured_loader.load() + pymupdf_loader.load()
                print("Ingested raw PDF documents using both loaders.")
            else:
                print(f"Failed to download PDF: {url}")
                continue
        else:
            loader = FireCrawlLoader(url=url, mode="scrape")
            raw_documents = loader.load()

        for doc in raw_documents:
            doc.page_content = merge_wrapped_rows(doc.page_content)
            doc.metadata = clean_metadata(doc.metadata)
            doc.metadata.update({
                "section": extract_section_heading(doc.page_content),
                "product_code": extract_product_code(doc.page_content),
                "source": url,
                "file_name": os.path.basename(url)
            })

        grouped_docs = group_table_related_docs(raw_documents)
        print(f"Grouped {len(raw_documents)} documents into {len(grouped_docs)} groups.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(grouped_docs)

        print(f"Going to add {len(documents)} document(s) to Pinecone Vector {INDEX_NAME}.")

        PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=INDEX_NAME
        )

        print(f"***Finished loading {url} to Pinecone vectorstore.***")

if __name__ == "__main__":
    ingest_docs()
    print("Ingestion completed.")
