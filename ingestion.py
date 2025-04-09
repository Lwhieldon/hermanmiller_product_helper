import fitz
import pytesseract
from PIL import Image
import os
import tempfile
import re
import logging
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "processed_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(filename="ingestion.log", level=logging.INFO, filemode="w",
                    format="%(asctime)s [%(levelname)s] %(message)s")

INDEX_NAME_2 = os.getenv("INDEX_NAME_2", "hermanmiller-product-helper-images")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

SAFE_LIMIT_BYTES = 39000

FEATURE_TERMS = [
    "microbecare", "surface material", "veneer", "glass", "edge options",
    "eased-edge", "thin-edge", "squared-edge", "sliding door", "bracket",
    "attachment", "paint finish", "textile", "coating", "surface option",
    "top cap", "screen", "panel", "grommet", "laminate"
]

def is_too_large(content: str, metadata: Dict[str, Any]) -> bool:
    combined_size = len(content.encode("utf-8")) + len(json.dumps(metadata).encode("utf-8"))
    return combined_size > SAFE_LIMIT_BYTES

def is_feature_page(text: str) -> bool:
    return any(term in text.lower() for term in FEATURE_TERMS)

def extract_text_with_ocr(page):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pix = page.get_pixmap(dpi=300)
        tmp.write(pix.tobytes("png"))
        tmp_path = tmp.name
    image = Image.open(tmp_path).convert("L").resize((pix.width * 2, pix.height * 2))
    os.remove(tmp_path)
    return pytesseract.image_to_string(image, config="--psm 6").strip()

def extract_top_header_text(page, dpi=300, crop_ratio=0.2) -> str:
    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    top_crop = image.crop((0, 0, pix.width, int(pix.height * crop_ratio)))
    top_crop = top_crop.resize((top_crop.width * 2, top_crop.height * 2))
    text = pytesseract.image_to_string(top_crop, config="--psm 6").strip()
    return text

def extract_part_numbers(text: str) -> List[str]:
    return list({match.lower() for match in re.findall(r"\b[A-Z]{2}\d{3,4}\b", text)})

def extract_pricing_table_as_markdown(text: str) -> str:
    import pandas as pd
    lines = text.splitlines()
    table_lines = [line for line in lines if re.search(r"\$\d", line)]
    rows = []
    for line in table_lines:
        line = re.sub(r"\s{2,}", "\t", line.strip())
        cells = re.split(r"\t+", line)
        price_cells = [c for c in cells if re.search(r"\$\d", c)]
        all_cells = cells[:len(cells) - len(price_cells)] + price_cells
        if len(all_cells) >= 2:
            rows.append(all_cells)
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    headers = ["Col" + str(i + 1) for i in range(max_cols)]
    for row in rows:
        while len(row) < max_cols:
            row.append("")
    df = pd.DataFrame(rows, columns=headers)
    return df.to_markdown(index=False)

def extract_image_metadata(doc, page_parts: Dict[int, List[str]], header_parts: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    image_data = []
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img_filename = f"page_{i+1:03d}_full.png"
        img_path = os.path.join(OUTPUT_DIR, img_filename)
        with open(img_path, "wb") as f:
            f.write(pix.tobytes("png"))

        ocr_parts = extract_part_numbers(extract_text_with_ocr(page))
        header_text = extract_top_header_text(page)
        header_parts_found = extract_part_numbers(header_text)

        nearby_parts = set(ocr_parts + header_parts_found)
        for offset in [-2, -1, 0, 1, 2]:
            p = page_parts.get(i + 1 + offset)
            if p:
                nearby_parts.update(p)

        image_data.append({
            "page": i + 1,
            "path": img_path,
            "caption": f"Illustration from page {i + 1}",
            "ocr_parts": ocr_parts,
            "header_parts": header_parts_found,
            "linked_parts": list(nearby_parts)
        })

        logging.info(f"🖼️ Image on page {i+1} linked to: {list(nearby_parts)}")

    return image_data

def split_large_chunk(content: str, metadata: dict, chunk_size: int = 1000) -> List[Document]:
    chunks = []
    for i in range(0, len(content), chunk_size):
        subtext = content[i:i + chunk_size]
        submeta = dict(metadata)
        submeta["chunk_index"] = i // chunk_size
        if not is_too_large(subtext, submeta):
            chunks.append(Document(page_content=subtext, metadata=submeta))
    return chunks

def extract_chunks_with_grouping(pdf_path: str) -> List[Document]:
    doc = fitz.open(pdf_path)
    chunks = []
    part_buffer = {}
    page_parts = {}
    header_parts = {}

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text().strip()
        header_text = extract_top_header_text(page)
        found_parts = extract_part_numbers(text)
        header_parts_found = extract_part_numbers(header_text)

        header_parts[i + 1] = header_parts_found

        if not found_parts and ("continued" in header_text.lower() or "continued" in text.lower()):
            found_parts = page_parts.get(i, []) or page_parts.get(i - 1, [])

        page_parts[i + 1] = found_parts

        if found_parts:
            for part in found_parts:
                part = part.lower()
                if part not in part_buffer:
                    part_buffer[part] = {"pages": [], "text": []}
                part_buffer[part]["pages"].append(i + 1)
                part_buffer[part]["text"].append(text)
        else:
            metadata = {"page": i + 1, "source": os.path.basename(pdf_path)}
            if is_feature_page(text):
                metadata["is_feature_block"] = True
            if not is_too_large(text, metadata):
                chunks.append(Document(page_content=text, metadata=metadata))

    for part, data in part_buffer.items():
        joined = "\n".join(data["text"])
        markdown = extract_pricing_table_as_markdown(joined)
        metadata = {
            "part_numbers": [part],
            "pages": ",".join(map(str, data["pages"][:10])),
            "source": os.path.basename(pdf_path),
            "is_pricing_table": True
        }
        content = markdown or joined
        if is_too_large(content, metadata):
            chunks.extend(split_large_chunk(content, metadata))
        else:
            chunks.append(Document(page_content=content, metadata=metadata))
        logging.info(f"💲 Added pricing chunk for {part}")

    images = extract_image_metadata(doc, page_parts, header_parts)
    for img in images:
        meta = {
            "page": img["page"],
            "image_path": os.path.basename(img["path"]),
            "caption": img["caption"],
            "source": os.path.basename(pdf_path),
            "part_numbers": img["linked_parts"][:15]
        }
        if not is_too_large("Image illustration", meta):
            chunks.append(Document(page_content="Image illustration", metadata=meta))
        else:
            chunks.extend(split_large_chunk("Image illustration", meta))
        logging.info(f"🔗 Image chunk added from page {img['page']}")

    return chunks

if __name__ == "__main__":
    pdf_path = "PB_CWB.pdf"
    docs = extract_chunks_with_grouping(pdf_path)

    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    if docs:
        total = 0
        for batch in chunked(docs, 100):
            PineconeVectorStore.from_documents(
                batch,
                embedding=embeddings,
                index_name=INDEX_NAME_2,
                text_key="page_content"
            )
            total += len(batch)
            logging.info(f"✅ Uploaded {total} chunks to Pinecone")
    else:
        logging.warning("❌ No chunks extracted.")
