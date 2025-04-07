import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
import tempfile
import re
import logging
import requests
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "processed_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(filename="ingestion.log", level=logging.INFO, filemode="w",
                    format="%(asctime)s [%(levelname)s] %(message)s")

INDEX_NAME = os.getenv("INDEX_NAME", "hermanmiller-product-helper")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def download_pdf(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.basename(url)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, "wb") as f:
            f.write(response.content)
        logging.info(f"PDF downloaded: {temp_path}")
        return temp_path
    else:
        raise Exception(f"Failed to download PDF: {url}")

def extract_text_with_ocr(page):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pix = page.get_pixmap(dpi=300)
        tmp.write(pix.tobytes("png"))
        tmp_path = tmp.name

    image = Image.open(tmp_path)
    text = pytesseract.image_to_string(image)
    os.remove(tmp_path)
    return text.strip()

def extract_images_and_illustrations(doc):
    image_data = []
    for i in range(len(doc)):
        page = doc[i]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = f"page_{i+1}_img{img_index+1}.{img_ext}"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            image_data.append({
                "page": i + 1,
                "path": img_path,
                "caption": f"Illustration on page {i+1}"
            })
    return image_data

def extract_part_numbers(text: str) -> List[str]:
    return list(set(re.findall(r"\b[A-Z]{2}\d{3,4}\b", text)))

def extract_price_values(text: str) -> List[str]:
    return re.findall(r"(?:\$)?\d{2,4}(?:\.\d{2})?", text)

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
        logging.info("No valid pricing rows found.")
        return ""
    max_cols = max(len(row) for row in rows)
    headers = ["Col" + str(i + 1) for i in range(max_cols)]
    for row in rows:
        while len(row) < max_cols:
            row.append("")
    df = pd.DataFrame(rows, columns=headers)
    logging.info(f"Extracted pricing table with {len(rows)} rows and {max_cols} columns")
    return df.to_markdown(index=False)

def extract_feature_blocks(text: str):
    features = []
    feature_keywords = [
        ("Surface Materials", r"(?i)surface material[s]?:?\s*(.*?)\n\n"),
        ("Edge Options", r"(?i)edge option[s]?:?\s*(.*?)\n\n"),
        ("MicrobeCare", r"(?i)microbecare.*?\n(.*?)(?=\n\n|\Z)"),
        ("Attachment Brackets", r"(?i)attachment bracket[s]?:?\s*(.*?)\n\n"),
        ("Top Cap Finish", r"(?i)top cap finish.*?\n(.*?)(?=\n\n|\Z)"),
        ("Glass Options", r"(?i)(?:screen finish|glass finish).*?\n(.*?)(?=\n\n|\Z)"),
        ("Fabric Grades", r"(?i)price category.*?\n(.*?)(?=\n\n|\Z)"),
        ("Veneer Options", r"(?i)veneer.*?\n(.*?)(?=\n\n|\Z)")
    ]
    for label, pattern in feature_keywords:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            content = match.strip()
            if content:
                features.append(Document(
                    page_content=f"{label} Options:\n{content}",
                    metadata={"feature_label": label, "is_feature_block": True}
                ))
                logging.info(f"Extracted feature block for {label}")
    return features

def extract_chunks_with_smart_grouping(pdf_path):
    import pandas as pd
    def extract_ft_price_table_with_finishes(text: str) -> str:
        base_pattern = re.compile(r"(FT\d{3})[.\s]+(\d{2})\s+(\d{2})\s*((?:[A-Z] \$?\d{2,4}(?:\.\d{2})?\s*)+)")
        finish_blocks = re.findall(r"(?i)(Metallic Paint.*?)\n(?:\s*\n|\Z)", text, re.DOTALL)
        rows = []
        for match in base_pattern.findall(text):
            part, height, width, price_block = match
            base_prices = dict(re.findall(r"([A-Z]) \$?(\d{2,4}(?:\.\d{2})?)", price_block))
            for opt, base in base_prices.items():
                rows.append([part, opt, "Standard", height, width, f"${float(base):.2f}"])
            for finish_block in finish_blocks:
                lines = finish_block.splitlines()
                for line in lines:
                    finish_info = re.findall(r"([A-Z]{2,3})\s+\+?\$?(\d+)", line)
                    for finish_code, upcharge in finish_info:
                        for opt, base in base_prices.items():
                            final_price = float(base) + float(upcharge)
                            rows.append([part, opt, finish_code, height, width, f"${final_price:.2f}"])
        if not rows:
            return ""
        df = pd.DataFrame(rows, columns=["Part", "Option", "Finish", "Height", "Width", "Price"])
        raw_finish_block = "\n".join(finish_blocks)
        return df.to_markdown(index=False) + "\n\nFinish Modifier Info:\n" + raw_finish_block

    logging.info(f"Processing file: {pdf_path}")
    doc = fitz.open(pdf_path)
    image_metadata = extract_images_and_illustrations(doc)
    chunks = []
    part_buffer = {}
    image_part_map = {}
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            text = extract_text_with_ocr(page)
        found_parts = extract_part_numbers(text)
        for img in image_metadata:
            if img["page"] == i + 1:
                image_part_map[img["path"]] = found_parts
        if found_parts:
            for part in found_parts:
                part_lower = part.lower()
                if part_lower not in part_buffer:
                    part_buffer[part_lower] = {"pages": [], "text": []}
                part_buffer[part_lower]["pages"].append(i + 1)
                part_buffer[part_lower]["text"].append(text)
        else:
            chunks.extend(extract_feature_blocks(text))
            meta = {"page": i + 1, "source": pdf_path}
            chunks.append(Document(page_content=text, metadata=meta))
    for part, data in part_buffer.items():
        merged_text = "\n".join(data["text"])
        markdown = extract_ft_price_table_with_finishes(merged_text) or extract_pricing_table_as_markdown(merged_text)
        meta = {
            "part_numbers": [part],
            "joined_parts": part,
            "pages": [str(p) for p in data["pages"]],
            "source": pdf_path,
            "is_pricing_table": True
        }
        chunks.append(Document(page_content=markdown or merged_text, metadata=meta))
        logging.info(f"Added pricing chunk for {part} with pages {data['pages']}")
    for img in image_metadata:
        related_parts = image_part_map.get(img["path"], [])
        chunks.append(Document(
            page_content="Image illustration",
            metadata={
                "page": img["page"],
                "image_path": img["path"],
                "caption": img["caption"],
                "source": pdf_path,
                "part_numbers": [p.lower() for p in related_parts] if related_parts else []
            }
        ))
        logging.info(f"Linked image on page {img['page']} to parts {related_parts}")
    logging.info(f"Finished processing {pdf_path}, total chunks: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    urls = [
        "https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf"
    ]
    all_docs = []
    for url in urls:
        try:
            pdf_path = download_pdf(url)
            docs = extract_chunks_with_smart_grouping(pdf_path)
            print(f"Extracted {len(docs)} chunks from {os.path.basename(pdf_path)}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {url}: {e}")
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]
    if all_docs:
        total = 0
        for batch in chunked(all_docs, 100):
            PineconeVectorStore.from_documents(
                batch,
                embedding=embeddings,
                index_name=INDEX_NAME,
                text_key="page_content"
            )
            total += len(batch)
            print(f"✅ Ingested {total} chunks so far...")
        print(f"✅ Finished ingesting all {total} chunks into Pinecone index '{INDEX_NAME}'")
    else:
        print("No documents to ingest.")