import os
import tempfile
import requests
import logging
import re
from typing import List
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LAParams
from pdfminer.image import ImageWriter

# Setup
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
INDEX_NAME = os.getenv("INDEX_NAME")

TEMP_IMAGE_DIR = os.path.join(tempfile.gettempdir(), "pdf_images")
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

logging.basicConfig(
    filename="ingestion.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def download_pdf(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(url))
        with open(temp_path, "wb") as f:
            f.write(response.content)
        logging.info(f"PDF downloaded: {temp_path}")
        return temp_path
    else:
        raise Exception(f"Failed to download PDF: {url}")


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool, str))}


def is_price(text: str) -> bool:
    return bool(re.search(r"\$\d+", text))


def extract_part_numbers(text: str) -> List[str]:
    matches = re.findall(r"\bFT\d{3,4}\b", text)
    return list(set(matches))



def merge_wrapped_lines(text: str) -> str:
    lines = text.split("\n")
    merged = []
    for line in lines:
        if merged and line.startswith(" "):
            merged[-1] += " " + line.strip()
        else:
            merged.append(line)
    return "\n".join(merged)


def save_image_from_ltimage(image_obj, page_num, index):
    image_writer = ImageWriter(TEMP_IMAGE_DIR)
    image_name = f"page_{page_num}_img_{index}.jpg"
    try:
        image_writer.export_image(image_obj, image_name)
        return os.path.join(TEMP_IMAGE_DIR, image_name)
    except Exception as e:
        logging.warning(f"Image extract fail (p{page_num}): {e}")
        return None


def extract_elements(pdf_path: str) -> List[dict]:
    elements = []
    laparams = LAParams()
    pages = list(extract_pages(pdf_path, laparams=laparams))
    for page_num, layout in enumerate(pages):
        image_index = 0
        for element in layout:
            if isinstance(element, LTTextContainer):
                text = merge_wrapped_lines(element.get_text()).strip()
                if not text:
                    continue
                typ = "heading" if element.height > 12 else ("price" if is_price(text) else "text")
                elements.append({"type": typ, "text": text, "page": page_num + 1})
            elif isinstance(element, LTImage):
                image_path = save_image_from_ltimage(element, page_num + 1, image_index)
                if image_path:
                    elements.append({
                        "type": "image",
                        "text": "Image",
                        "page": page_num + 1,
                        "image_path": image_path
                    })
                    image_index += 1
    return elements


def chunk_intelligently(elements: List[dict], global_metadata: dict) -> List[Document]:
    chunks = []
    current_chunk_text = ""
    current_chunk_metadata = global_metadata.copy()

    last_part_numbers = []
    last_heading = ""

    def finish_chunk():
        nonlocal current_chunk_text, current_chunk_metadata, last_part_numbers
        if current_chunk_text.strip():
            part_numbers = extract_part_numbers(current_chunk_text)
            if part_numbers:
                current_chunk_metadata["part_numbers"] = part_numbers
                last_part_numbers = part_numbers
                logging.info(f"Chunk FT codes: {part_numbers}")
            if "$" in current_chunk_text:
                current_chunk_metadata["contains_pricing_table"] = True
            chunks.append(Document(page_content=current_chunk_text.strip(), metadata=current_chunk_metadata.copy()))
        current_chunk_text = ""
        current_chunk_metadata = global_metadata.copy()

    for element in elements:
        text = element["text"]
        page = element["page"]
        extracted_parts = extract_part_numbers(text)
        is_short_snippet = len(text.strip()) <= 150 and extracted_parts

        if element["type"] == "heading":
            finish_chunk()
            last_heading = text
            current_chunk_metadata["heading"] = text
            current_chunk_metadata["page"] = page
            current_chunk_text = text

        elif element["type"] == "image":
            image_meta = {
                **global_metadata,
                "type": "image",
                "image_path": element["image_path"],
                "page": page
            }
            if last_heading:
                image_meta["prev_heading"] = last_heading
            if last_part_numbers:
                image_meta["part_numbers"] = last_part_numbers
                logging.info(f"Image at page {page} linked to FT: {last_part_numbers}")
            chunks.append(Document("Image related to product", metadata=image_meta))

        elif is_short_snippet or extracted_parts:
            snippet_meta = {
                **global_metadata,
                "page": page
            }
            if extracted_parts:
                snippet_meta["part_numbers"] = extracted_parts
                logging.info(f"✅ FT codes from text (p{page}): {extracted_parts}")
            chunks.append(Document(text.strip(), metadata=snippet_meta))
            logging.info(f"Indexed FT snippet on page {page}: {extracted_parts}")

        elif len(current_chunk_text) + len(text) + 1 <= 400:
            current_chunk_text += " " + text
            current_chunk_metadata["page"] = page
        else:
            finish_chunk()
            current_chunk_text = text
            current_chunk_metadata["page"] = page

    finish_chunk()
    return chunks


def process_pdf(pdf_path: str) -> List[Document]:
    raw_metadata = {"source": pdf_path}
    chunks = []
    try:
        elements = extract_elements(pdf_path)
        chunks = chunk_intelligently(elements, raw_metadata)
        for i, chunk in enumerate(chunks):
            prev_heading = None
            for j in range(i - 1, -1, -1):
                if "heading" in chunks[j].metadata:
                    prev_heading = chunks[j].metadata["heading"]
                    break
            chunk.metadata["prev_heading"] = prev_heading
    except Exception as e:
        logging.error(f"PDF processing error: {e}")
    return chunks


def ingest_docs():
    urls = [
        "https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf"
    ]

    all_docs = []
    for url in urls:
        try:
            pdf_path = download_pdf(url)
            docs = process_pdf(pdf_path)
            for doc in docs:
                doc.metadata["source_title"] = os.path.basename(url)
            all_docs.extend(docs)
        except Exception as e:
            logging.error(f"Ingestion failed for {url}: {e}")

    filtered_docs = [doc for doc in all_docs if doc.page_content and (
    len(doc.page_content.strip()) > 10 or "part_numbers" in doc.metadata)]

    for doc in filtered_docs:
        doc.metadata = clean_metadata(doc.metadata)

    PineconeVectorStore.from_documents(
        documents=filtered_docs,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    logging.info(f"✅ {len(filtered_docs)} chunks indexed into Pinecone.")


if __name__ == "__main__":
    ingest_docs()
    logging.info("🎉 Ingestion complete.")
