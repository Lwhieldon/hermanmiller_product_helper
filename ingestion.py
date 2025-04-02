from dotenv import load_dotenv
import nltk
import os
import tempfile
import requests
import re
import base64
import uuid
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.document_loaders import (
    FireCrawlLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    PDFPlumberLoader
)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
chat = ChatOpenAI(model="gpt-4", temperature=0)

# --- Util Functions ---

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
    price_line_count = sum(1 for line in lines if re.search(r"\$\d{2,}", line))
    ft_code_count = sum(1 for line in lines if re.search(r"\bFT\d{3}\.", line))
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
    return re.findall(r"Step \d+\.\s*(.*?)\n", text)

def extract_section_heading(text):
    patterns = ["Walls", "Work Surfaces", "Storage", "Screens", "Lighting"]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return "General"

def merge_wrapped_rows(text):
    lines = text.split("\n")
    merged_lines = []
    for line in lines:
        if merged_lines and line.startswith(" "):
            merged_lines[-1] += " " + line.strip()
        else:
            merged_lines.append(line)
    return "\n".join(merged_lines)

def clean_metadata(metadata):
    return {k: v for k, v in metadata.items() if v is not None}

def split_by_product_code(docs):
    grouped = []
    current_chunk = []
    current_code = None
    for doc in docs:
        lines = doc.page_content.splitlines()
        for line in lines:
            match = re.match(r"^(FT\d{3}\.)", line)
            if match:
                if current_chunk:
                    grouped.append(Document(
                        page_content="\n".join(current_chunk),
                        metadata={**doc.metadata, "product_code": current_code}
                    ))
                    current_chunk = []
                current_code = match.group(1)
            current_chunk.append(line)
        if current_chunk:
            grouped.append(Document(
                page_content="\n".join(current_chunk),
                metadata={**doc.metadata, "product_code": current_code}
            ))
            current_chunk = []
    return grouped

def generate_element_id(content):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content[:50]))

def call_openai_to_structurize_table(text_block, max_retries=2):
    def generate_prompt(table_text):
        return f"""
You are a product catalog assistant.

Extract all possible permutations of product configuration options from the following price table block. Each permutation should include all relevant attributes (e.g., height, width, depth, base option letter) and its price.

**Instructions**:
- Only output a single valid JSON object.
- Do NOT add explanations, comments, or anything outside the JSON.
- Format the output exactly like the example below.
- The input table is between triple backticks.

Example Output:
{{
  "product_code": "FT110",
  "options": [
    {{"height": "35", "width": "18", "option": "A", "price": 219}},
    {{"height": "35", "width": "18", "option": "N", "price": 246}},
    {{"height": "35", "width": "18", "option": "J", "price": 0}},
    {{"height": "35", "width": "18", "option": "R", "price": 0}},
    {{"height": "35", "width": "18", "option": "X", "price": 203}},
    {{"height": "35", "width": "24", "option": "A", "price": 232}},
    {{"height": "35", "width": "24", "option": "N", "price": 264}}
  ]
}}

Input Table:
```{table_text}```
"""

    last_response = None

    for attempt in range(max_retries + 1):
        prompt = generate_prompt(text_block)
        response = chat.invoke(prompt)
        last_response = response.content

        try:
            json_text_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_text_match:
                return json.loads(json_text_match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "raw_table": text_block,
        "error": f"Failed to parse JSON after {max_retries + 1} attempts",
        "raw_response": last_response
    }

def initialize_reasoning_agent(documents):
    def query_table_tool(input):
        query = input.lower()
        results = []
        for doc in documents:
            if "structured_price_json" in doc.metadata:
                data = doc.metadata["structured_price_json"]
                product_code = data.get("product_code", "")
                options = data.get("options", [])
                for opt in options:
                    match_product = product_code.lower() in query if product_code else False
                    match_option = str(opt.get("option", "")).lower() in query if "option" in opt else True
                    match_height = str(opt.get("height", "")).lower() in query if "height" in opt else True
                    match_width = str(opt.get("width", "")).lower() in query if "width" in opt else True
                    if match_product and match_option and match_height and match_width:
                        results.append(opt)

        if "cheapest" in query and results:
            results = sorted(results, key=lambda x: x.get("price", float('inf')))
            return json.dumps(results[:1], indent=2)
        return json.dumps(results, indent=2)

    tools = [
        Tool(
            name="TableQuery",
            func=query_table_tool,
            description="Use this to answer questions about product prices and configurations, including filtering by base option."
        )
    ]
    return initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

def ingest_docs():
    INDEX_NAME = os.getenv("INDEX_NAME")
    documents_base_urls = [
        "https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf"
    ]

    for url in documents_base_urls:
        print(f"Loading {url}")
        if url.endswith(".pdf"):
            pdf_path = download_pdf(url)
            if not pdf_path:
                continue

            pdfplumber_loader = PDFPlumberLoader(pdf_path)
            pymupdf_loader = PyMuPDFLoader(pdf_path)
            unstructured_loader = UnstructuredPDFLoader(
                file_path=pdf_path,
                mode="elements",
                strategy="hi_res",
                unstructured_kwargs={"detect_tables": True, "use_image": True, "include_metadata": True}
            )
            raw_documents = (
                pdfplumber_loader.load()
                + pymupdf_loader.load()
                + unstructured_loader.load()
            )
        else:
            loader = FireCrawlLoader(url=url, mode="scrape")
            raw_documents = loader.load()

        for doc in raw_documents:
            doc.page_content = merge_wrapped_rows(doc.page_content)
            doc.metadata = clean_metadata(doc.metadata)

            structured_prices = extract_all_prices(doc.page_content)
            if structured_prices:
                table_txt = "\n".join([" | ".join(structured_prices["columns"])]+ structured_prices["rows"])
                doc.page_content += f"\n\n[PRICE TABLE]\n{table_txt}"
                structured_json = call_openai_to_structurize_table(table_txt)
                doc.metadata["structured_price_json"] = structured_json

            page_number = doc.metadata.get("page")
            prod_code = extract_product_code(doc.page_content)

            enriched = {
                "section": extract_section_heading(doc.page_content),
                "product_code": prod_code,
                "product_name": extract_product_name_top_of_page(doc.page_content),
                "price_table_present": bool(structured_prices),
                "all_prices": structured_prices,
                "spec_steps": extract_spec_steps(doc.page_content),
                "source": url,
                "file_name": os.path.basename(url),
                "page_number": page_number,
                "coordinates_layout_width": 1700,
                "coordinates_layout_height": 2200,
                "coordinates_system": "PixelSpace",
                "element_id": generate_element_id(doc.page_content)
            }
            doc.metadata.update(clean_metadata(enriched))

        grouped_docs = split_by_product_code(raw_documents)
        print(f"Total chunks: {len(grouped_docs)}")

        documents = [doc for doc in grouped_docs if len(doc.page_content.strip()) > 100]

        for doc in documents:
            doc.metadata = clean_metadata(doc.metadata)

        for d in documents[:3]:
            print(f"\n\n--- Preview Document ---\n{d.metadata}\n{d.page_content[:400]}\n---")

        PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=INDEX_NAME
        )

        agent = initialize_reasoning_agent(documents)
        result = agent.run("What's the price of a 42\" wide, 53\" high FT110 frame?")
        print("\n--- Agent Answer ---\n", result)

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
