from dotenv import load_dotenv
import nltk
import os
import tempfile
import requests
import re
import base64

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    FireCrawlLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader
)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def extract_product_name_top_of_page(text):
    lines = text.strip().split("\n")
    for line in lines[:5]:
        if re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$", line.strip()) and len(line.strip()) > 3:
            return line.strip()
    return None


def extract_product_code(text):
    match = re.search(r"(FT\d{3}\.)", text)
    return match.group(1) if match else None


def is_price_block(text):
    lines = text.splitlines()
    price_line_count = 0
    ft_code_count = 0

    for line in lines:
        if re.search(r"\$\d{2,}", line):
            price_line_count += 1
        if re.search(r"\bFT\d{3}\.", line):
            ft_code_count += 1

    return price_line_count >= 3 or ft_code_count >= 3


def extract_all_prices(text):
    lines = text.splitlines()
    structured_table = []
    headers = []
    capture = False
    product_line_found = False
    blank_line_count = 0

    for line in lines:
        if re.match(r"\s*[A-Z](?:\s+[A-Z])+", line):
            headers = re.findall(r"[A-Z]+", line)
            capture = True
            continue

        if capture:
            if re.search(r"\bFT\d{3}\.", line):
                product_line_found = True
                structured_table.append(line.strip())
                blank_line_count = 0
            elif product_line_found and re.search(r"\$\d{2,}", line):
                structured_table.append(line.strip())
                blank_line_count = 0
            elif product_line_found and line.strip() == "":
                blank_line_count += 1
                if blank_line_count > 1:
                    break
            elif product_line_found and not re.search(r"\$\d{2,}", line):
                break

    rows = [row for row in structured_table if row.strip() != ""]
    return {"columns": headers, "rows": rows} if rows else None


def extract_spec_steps(text):
    steps = re.findall(r"Step \d+\.\s*(.*?)\n", text)
    return steps if steps else None


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


def clean_metadata(metadata):
    flat = flatten_dict(metadata)
    return {k: v for k, v in flat.items() if v is not None}


def merge_wrapped_rows(text):
    lines = text.split("\n")
    merged_lines = []
    for line in lines:
        if merged_lines and line.startswith(" "):
            merged_lines[-1] += " " + line.strip()
        else:
            merged_lines.append(line)
    return "\n".join(merged_lines)


def extract_section_heading(text):
    SECTION_PATTERNS = [
        r"Walls", r"Work Surfaces", r"Storage", r"Screens", r"Lighting",
        r"Installation", r"Warranty", r"Connectors", r"Terms and Conditions"
    ]
    for pattern in SECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return "General"


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


def encode_image_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


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
                print("Ingested raw PDF documents using both loaders. Total length:", len(raw_documents))
            else:
                print(f"Failed to download PDF: {url}")
                continue
        else:
            loader = FireCrawlLoader(url=url, mode="scrape")
            raw_documents = loader.load()

        for doc in raw_documents:
            doc.page_content = merge_wrapped_rows(doc.page_content)
            doc.metadata = clean_metadata(doc.metadata)

            structured_prices = extract_all_prices(doc.page_content)
            if structured_prices:
                pretty_table = (
                    " | ".join(structured_prices["columns"]) + "\n" +
                    "\n".join(structured_prices["rows"])
                )
                doc.page_content += f"\n\n[PRICE TABLE]\n{pretty_table}"

            enriched = {
                "section": extract_section_heading(doc.page_content),
                "product_code": extract_product_code(doc.page_content),
                "product_name": extract_product_name_top_of_page(doc.page_content),
                "price_table_present": is_price_block(doc.page_content),
                "all_prices": structured_prices,
                "spec_steps": extract_spec_steps(doc.page_content),
                "source": url,
                "file_name": os.path.basename(url)
            }

            image_paths = doc.metadata.get("image_path", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            base64_images = [encode_image_base64(img) for img in image_paths if encode_image_base64(img)]
            if base64_images:
                enriched["images"] = base64_images

            doc.metadata.update({k: v for k, v in enriched.items() if v is not None})

        grouped_docs = group_table_related_docs(raw_documents)
        print(f"Total number of documents to process is {len(grouped_docs)}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(grouped_docs)

        print(f"Going to add {len(documents)} document(s) to Pinecone Vector Index {INDEX_NAME}")

        PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=INDEX_NAME
        )

        print(f"***Finished loading {url} to Pinecone vectorstore.***")


def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, os.path.basename(url))
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    return None


if __name__ == "__main__":
    ingest_docs()
    print("Ingestion completed.")
