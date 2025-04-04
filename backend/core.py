from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List, Optional  # Import Optional
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = "hermanmiller-product-helper"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(search_kwargs={"k": 5}), prompt=rephrase_prompt  # Increased k
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke({"input": query, "chat_history": chat_history, "question": query}) # Added "question" to the input
    return result


def process_response(llm_response: dict) -> dict:
    """
    Processes the LLM response to format it for display.
    This includes adding source information and potentially
    formatting the answer.
    """

    sources = []
    for source in llm_response.get("source_documents", []):  # Use .get() with a default value
        if hasattr(source, "metadata"):  # Check if source has metadata
            source_data = {
                "page": source.metadata.get("page", "Unknown"),
                "source": source.metadata.get("source", "Unknown"),
                "heading": source.metadata.get("heading", None),
                "prev_heading": source.metadata.get("prev_heading", None),
            }
            sources.append(source_data)
        else:
            print(f"Warning: source_document missing metadata: {source}")  # Log a warning

    return {
        "answer": llm_response.get("answer", ""),  # Safely get the answer
        "sources": sources,
    }


if __name__ == "__main__":
    query = "Tell me what G1 means?"
    llm_response = run_llm(query)
    print(llm_response)
    processed_response = process_response(llm_response)
    print(processed_response)