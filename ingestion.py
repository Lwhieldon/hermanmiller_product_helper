import os
import tempfile
import requests
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
import re
from collections import defaultdict

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTTextContainer,
    LTTextBoxHorizontal,
    LTImage,
    LTFigure,
    LAParams,
)

# Configure logging
logging.basicConfig(
    filename="ingestion.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load environment variables
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
INDEX_NAME = os.getenv("INDEX_NAME")


# --- Utility Functions ---

def download_pdf(url: str) -> str | None:
    response = requests.get(url)
    if response.status_code == 200:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, os.path.basename(url))
        with open(file_path, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded PDF to {file_path}")
        return file_path
    logging.warning(
        f"Failed to download PDF from {url} (status code: {response.status_code})"
    )
    return None


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}


def merge_wrapped_lines(text: str) -> str:
    lines = text.split("\n")
    merged = []
    for line in lines:
        if merged and line.startswith(" "):
            merged[-1] += " " + line.strip()
        else:
            merged.append(line)
    return "\n".join(merged)


def is_price(text: str) -> bool:
    """Very basic check if a string looks like a price."""
    return bool(re.match(r"^\$\d+\.?\d{0,2}$", text.strip()))


def chunk_intelligently(elements: List[dict], global_metadata: dict) -> List[Document]:
    MAX_CHUNK_LENGTH = 400  # Reduced chunk size
    MIN_CHUNK_LENGTH = 50
    chunks: List[Document] = []
    current_chunk_text = ""
    current_chunk_metadata = global_metadata.copy()

    def finish_chunk():
        nonlocal current_chunk_text, current_chunk_metadata  # Declare nonlocal
        if (
            current_chunk_text and len(current_chunk_text) > MIN_CHUNK_LENGTH
        ):  # Only add if it's long enough
            chunks.append(
                Document(
                    page_content=current_chunk_text.strip(),
                    metadata=current_chunk_metadata.copy(),  # Copy metadata for the chunk
                )
            )
        current_chunk_text = ""
        current_chunk_metadata = global_metadata.copy()  # Reset metadata to global_metadata

    for element in elements:
        text = element["text"]
        if not text:
            continue  # Skip empty elements

        if element["type"] == "heading":
            finish_chunk()
            current_chunk_metadata["heading"] = text
            current_chunk_text = text
        elif len(current_chunk_text) + len(text) + 1 <= MAX_CHUNK_LENGTH:  # Keep chunks smaller
            current_chunk_text += " " + text
        else:
            finish_chunk()
            current_chunk_text = text

    finish_chunk()  # Finish the last chunk
    return chunks


def extract_elements(pdf_path: str) -> List[dict]:
    """
    Extracts structured information from the PDF, handling columns and tables.
    """

    elements: List[Dict[str, Any]] = []
    try:
        laparams = LAParams(
            line_overlap=0.5,
            char_margin=2.0,
            line_margin=0.5,
            word_margin=0.1,
            boxes_flow=0.5,
            all_texts=False,
            detect_vertical=False,
        )

        pages = list(extract_pages(pdf_path, laparams=laparams))
        for page_num, page_layout in enumerate(pages):
            vertical_groups = defaultdict(list)
            for element in page_layout:
                y0 = round(element.y0, 2)
                vertical_groups[y0].append(element)

            for group in vertical_groups.values():
                group.sort(key=lambda x: x.x0)

            for group in vertical_groups.values():
                for element in group:
                    if isinstance(element, LTTextContainer):
                        text = merge_wrapped_lines(element.get_text()).strip()
                        if not text:
                            continue

                        if element.height > 12:
                            elements.append({"type": "heading", "text": text, "page": page_num + 1})
                        elif is_price(text):
                            elements.append({"type": "price", "text": text, "page": page_num + 1})
                        else:
                            elements.append({"type": "text", "text": text, "page": page_num + 1})
                    elif isinstance(element, LTImage):
                        elements.append({"type": "image", "text": "Image", "page": page_num + 1})
                    elif isinstance(element, LTFigure):
                        elements.append({"type": "figure", "text": "Figure", "page": page_num + 1})

    except Exception as e:
        logging.error(f"Error extracting elements from {pdf_path}: {e}")
    return elements


def process_pdf(pdf_path: str) -> List[Document]:
    """
    Processes a single PDF to extract, chunk, and prepare documents.
    """

    raw_metadata = {"source": pdf_path}
    chunks: List[Document] = []

    try:
        extracted_elements = extract_elements(pdf_path)
        chunks = chunk_intelligently(extracted_elements, raw_metadata)

        for i, chunk in enumerate(chunks):  # Add prev_heading to all chunks
            prev_heading = None
            for j in range(i - 1, -1, -1):
                if "heading" in chunks[j].metadata:
                    prev_heading = chunks[j].metadata["heading"]
                    break
            chunks[i].metadata["prev_heading"] = prev_heading
            chunks[i].metadata["page"] = extracted_elements[0]["page"] if extracted_elements else 1

    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return []

    return chunks


def ingest_docs():
    urls = [
        "https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf"
    ]

    all_docs: List[Document] = []

    for url in urls:
        logging.info(f"Processing {url}")
        pdf_path = download_pdf(url)
        if not pdf_path:
            logging.warning(f"Skipping {url} due to download failure")
            continue

        all_docs.extend(process_pdf(pdf_path))

    logging.info(f"Loaded {len(all_docs)} raw pages from PDF(s)")

    min_length = 50
    filtered_docs = []
    discarded_count = 0
    filter_patterns = [
        r"(HermanMiller|Appendix|Index|Page|Prices|continued|\d+ of \d+)",
        r"^\s*$",
        r"^\d{1,2}\/\d{1,2}\/\d{2,4}$" # added date pattern to filter
    ]

    for doc in all_docs:
        content = doc.page_content.strip()
        if len(content) > min_length and not any(
            re.search(pattern, content, re.IGNORECASE) for pattern in filter_patterns
        ):
            filtered_docs.append(doc)
        else:
            discarded_count += 1
            logging.debug(
                f'Filtered out short or irrelevant chunk ({len(content)} chars): "{content[:40]}..."'
            )

    split_docs = filtered_docs
    logging.info(f"Discarded {discarded_count} short or irrelevant chunks")
    logging.info(f"Retained {len(split_docs)} chunks for indexing")

    for doc in split_docs:
        doc.metadata = clean_metadata(doc.metadata)

    PineconeVectorStore.from_documents(
        documents=split_docs, embedding=embeddings, index_name=INDEX_NAME
    )

    logging.info(f"Indexed {len(split_docs)} chunks to Pinecone index '{INDEX_NAME}'")


# --- Main ---

if __name__ == "__main__":
    ingest_docs()
    logging.info("Ingestion completed")