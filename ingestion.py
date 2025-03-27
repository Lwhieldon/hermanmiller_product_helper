from dotenv import load_dotenv

load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import FireCrawlLoader, PyMuPDFLoader
import os
import tempfile
import requests

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, os.path.basename(url))
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    return None

def ingest_docs():
    INDEX_NAME = os.getenv("INDEX_NAME")
    langchain_documents_base_urls = [
   "https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf"
    ]
    
    for url in langchain_documents_base_urls:
        print(f"Loading {url}")
        
        if url.endswith(".pdf"):
            pdf_path = download_pdf(url)
            if pdf_path:
                loader = PyMuPDFLoader(pdf_path)
                raw_documents = loader.load()
            else:
                print(f"Failed to download PDF: {url}")
                continue
        else:
            loader = FireCrawlLoader(url=url, mode="scrape")
            raw_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)
        print(f"Going to add {len(raw_documents)} document(s) to Pinecone Vector {INDEX_NAME}.")
        
        for doc in documents:
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