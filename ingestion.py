# ingestion.py

import os
import tempfile
import requests
import logging
import re
from typing import List, Dict
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pdf2image import convert_from_path
import pytesseract
# from PIL import Image

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTImage, LAParams
from pdfminer.image import ImageWriter

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
INDEX_NAME = os.getenv("INDEX_NAME", "hermanmiller-product-helper")

TEMP_IMAGE_DIR = os.path.join(tempfile.gettempdir(), "pdf_images")
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

logging.basicConfig(filename="ingestion.log", level=logging.INFO, filemode="w",
                    format="%(asctime)s [%(levelname)s] %(message)s")


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


def is_price(text: str) -> bool:
    return bool(re.search(r"\d{2,4}(?:\.\d{2})?", text))


def extract_price_values(text: str) -> List[str]:
    return re.findall(r"(?:\$)?\d{2,4}(?:\.\d{2})?", text)


def extract_part_numbers(text: str) -> List[str]:
    matches = re.findall(r"\bFT\d{3,4}\b", text)
    return list(set(matches))


def extract_part_descriptions(text: str) -> List[Dict[str, str]]:
    pattern = re.findall(r"([A-Za-z0-9 \-\/]+?)\s*\(?\b(FT\d{3,4})\)?", text)
    return [{"part_number": pn, "description": desc.strip()} for desc, pn in pattern]


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
        logging.warning(f"Failed to save image on page {page_num}: {e}")
        return None


def extract_elements_with_ocr(pdf_path: str) -> List[dict]:
    elements = []
    laparams = LAParams()
    pages = list(extract_pages(pdf_path, laparams=laparams))
    images = convert_from_path(pdf_path, dpi=300)

    for page_idx, layout in enumerate(pages):
        page_num = page_idx + 1
        has_text = False
        image_index = 0
        text_blocks = []

        for element in layout:
            if isinstance(element, LTTextContainer):
                text = merge_wrapped_lines(element.get_text()).strip()
                if text:
                    has_text = True
                    text_blocks.append(text)
                    typ = "heading" if element.height > 12 else ("price" if is_price(text) else "text")
                    elements.append({"type": typ, "text": text, "page": page_num})

        if not has_text:
            ocr_text = pytesseract.image_to_string(images[page_idx])
            for line in ocr_text.splitlines():
                line = line.strip()
                if line:
                    typ = "price" if is_price(line) else "ocr"
                    text_blocks.append(line)
                    elements.append({"type": typ, "text": line, "page": page_num})

        page_text = " ".join(text_blocks)
        page_part_numbers = extract_part_numbers(page_text)

        for element in layout:
            if isinstance(element, LTImage):
                image_path = save_image_from_ltimage(element, page_num, image_index)
                if image_path:
                    elements.append({
                        "type": "image",
                        "text": "Image",
                        "page": page_num,
                        "image_path": image_path,
                        "linked_parts": page_part_numbers
                    })
                    image_index += 1

    return elements


# ingestion.py (partial, focusing on chunk_intelligently fix)

def chunk_intelligently(elements: List[dict], global_metadata: dict) -> List[Document]:
    chunks = []
    current_chunk_text = ""
    current_chunk_metadata = global_metadata.copy()

    last_part_numbers = []
    last_description = ""
    last_heading = ""

    def finish_chunk():
        nonlocal current_chunk_text, current_chunk_metadata, last_part_numbers
        if current_chunk_text.strip():
            part_numbers = extract_part_numbers(current_chunk_text)
            if part_numbers:
                current_chunk_metadata["part_numbers"] = part_numbers
                last_part_numbers = part_numbers

            desc_map = extract_part_descriptions(current_chunk_text)
            if desc_map:
                if len(desc_map) == 1:
                    current_chunk_metadata["description"] = desc_map[0]["description"]
                    last_description = desc_map[0]["description"]
                else:
                    all_descriptions = list({d["description"] for d in desc_map})
                    current_chunk_metadata["descriptions"] = all_descriptions[:5]

            price_lines = [line for line in current_chunk_text.splitlines() if len(extract_price_values(line)) >= 3]
            if price_lines:
                current_chunk_metadata["contains_pricing_table"] = True
                all_prices = []
                for line in price_lines:
                    all_prices.extend(extract_price_values(line))
                current_chunk_metadata["price_values"] = list(set(all_prices))

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
                "page": page,
            }
            if last_heading:
                image_meta["prev_heading"] = last_heading
            if last_description:
                image_meta["description"] = last_description

            if "linked_parts" in element:
                image_meta["part_numbers"] = list(set(element["linked_parts"]))
            elif last_part_numbers:
                image_meta["part_numbers"] = list(set(last_part_numbers))

            if image_meta.get("part_numbers"):
                logging.info(f"Tagged image on page {page} with parts: {image_meta['part_numbers']}")

            part_info = image_meta.get("part_numbers", [])
            desc = image_meta.get("description", "") or image_meta.get("prev_heading", "") or "unknown"
            content = f"Illustration for {', '.join(part_info)}: {desc}" if part_info else f"Product illustration: {desc}"

            chunks.append(Document(page_content=content, metadata=image_meta))

        elif is_short_snippet or extracted_parts:
            snippet_meta = {
                **global_metadata,
                "page": page
            }
            if extracted_parts:
                snippet_meta["part_numbers"] = extracted_parts
                last_part_numbers = extracted_parts

                desc_map = extract_part_descriptions(text)
                if desc_map:
                    if len(desc_map) == 1:
                        snippet_meta["description"] = desc_map[0]["description"]
                        last_description = desc_map[0]["description"]
                    else:
                        all_descriptions = list({d["description"] for d in desc_map})
                        snippet_meta["descriptions"] = all_descriptions[:5]

                logging.info(f"Indexed part {extracted_parts} with desc on page {page}: {desc_map}")
            chunks.append(Document(text.strip(), metadata=snippet_meta))

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
        elements = extract_elements_with_ocr(pdf_path)
        chunks = chunk_intelligently(elements, raw_metadata)
        for i, chunk in enumerate(chunks):
            for j in range(i - 1, -1, -1):
                if "heading" in chunks[j].metadata:
                    chunk.metadata["prev_heading"] = chunks[j].metadata["heading"]
                    break
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
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
            all_docs.extend(docs)
        except Exception as e:
            logging.error(f"Error ingesting from {url}: {e}")

    if all_docs:
        vectorstore = PineconeVectorStore.from_documents(all_docs, embedding=embeddings, index_name=INDEX_NAME)
        print(f"{len(all_docs)} documents ingested into Pinecone '{INDEX_NAME}' of {vectorstore}")
    else:
        print("No documents ingested.")


if __name__ == "__main__":
    try:
        ingest_docs()
    except Exception as e:
        print(f"Remote ingestion failed: {e}")
