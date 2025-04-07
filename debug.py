from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
import os

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(index_name="hermanmiller-product-helper", embedding=embedding)

docs = vectorstore.similarity_search("FT123", k=50)

print("FT123 Image Chunks:")
for doc in docs:
    meta = doc.metadata
    if meta.get("image_path"):
        print(f"- Image Path: {meta['image_path']}")
        print(f"  Page: {meta.get('page')}")
        print(f"  Caption: {meta.get('description') or meta.get('prev_heading')}")
        print(f"  Part Numbers: {meta.get('part_numbers')}")
        print()
