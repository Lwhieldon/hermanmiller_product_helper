from dotenv import load_dotenv
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import FireCrawlLoader, PyMuPDFLoader, UnstructuredPDFLoader
import os
import tempfile
import requests

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, os.path.basename(url))
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    return None

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
    return flatten_dict(metadata)

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
                loader = UnstructuredPDFLoader(
                    file_path=pdf_path, 
                    mode="elements", 
                    strategy="hi_res",
                    unstructured_kwargs={
                        "detect_tables": True,  # Enable table detection
                        "use_image": True,  # Enable image extraction if needed
                        "image_quality": "high",  # Set image quality to high 
                        "include_metadata": True,  # Include metadata in the output
                        "image_processing": "auto",  # Automatically process images for better quality
                        "include_text": True,  # Ensure text extraction is enabled
                        "text_quality": "high",  # Set text extraction quality to high
                        "include_page_breaks": True,  # Include page breaks in the output
                        "page_break_strategy": "auto",  # Automatically determine page break strategy
                        "ocr_languages": ["eng"]  # Specify languages for OCR if needed                
                    }
                    )               
                raw_documents = loader.load()
                print("ingested raw pdf documents from UnstructuredPDFLoader.")
            else:
                print(f"Failed to download PDF: {url}")
                continue
        else:
            loader = FireCrawlLoader(url=url, mode="scrape")
            raw_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n", " ", ""])
        documents = text_splitter.split_documents(raw_documents)
        print(f"Going to add {len(raw_documents)} document(s) to Pinecone Vector {INDEX_NAME}.")
        
        for doc in documents:
            doc.metadata = clean_metadata(doc.metadata)
            doc.metadata.update({"source": url})
        
        PineconeVectorStore.from_documents(
            documents, 
            embeddings, 
            index_name="hermanmiller-product-helper"
        )
        print(f"***Finished loading {url} to Pinecone vectorstore.***")
                
if __name__ == "__main__":
    ingest_docs()
    print("Ingestion completed.")
