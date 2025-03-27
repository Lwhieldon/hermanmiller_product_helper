# Herman Miller Product Helper Bot 🪑🤖

An AI-powered assistant for Herman Miller product data, combining **LangChain**, **OpenAI**, **Pinecone**, and **Streamlit**. This chatbot allows users to ask product-related questions and receive intelligent, citation-backed answers from ingested PDFs and webpages.

---

## 📂 Project Structure

| File           | Description |
|----------------|-------------|
| `ingestion.py` | Fetches, chunks, embeds, and stores product documents into Pinecone. |
| `main.py`      | Streamlit frontend to chat with the AI assistant and display responses. |
| `core.py`      | Backend logic that retrieves documents and runs an LLM to generate answers. |

---

## 🚀 Getting Started

### 1. Install Dependencies (Using Pipenv)

This project uses **Pipenv** for managing dependencies and virtual environments.

First, install Pipenv (if you don't have it yet):

```bash
pip install pipenv
```

Then, install the environment from the `Pipfile` and `Pipfile.lock`:

```bash
pipenv install
```

To activate the virtual environment:

```bash
pipenv shell
```

---

### 2. Set Up Environment Variables

Create a `.env` file in your project root with the following:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
INDEX_NAME=hermanmiller-product-helper
```

---

### 3. Ingest Documents

To populate the Pinecone vector index:

```bash
python ingestion.py
```

This:

- Downloads or scrapes product pages (currently a Herman Miller PDF),
- Splits the content into chunks,
- Embeds them using OpenAI’s `text-embedding-3-small`,
- Uploads them to Pinecone.

---

### 4. Run the App

Launch the chatbot frontend:

```bash
streamlit run main.py
```

You’ll get:

- A sidebar with a profile (Gravatar support),
- A clean interface for chatting with the AI bot,
- Instant answers with source citations.

---

## 🧠 How It Works

### 🗂 `ingestion.py`

- Loads Herman Miller product documents (e.g., PDFs),
- Splits them into semantic chunks,
- Embeds and stores them in Pinecone.

### 🧠 `core.py`

- Uses LangChain's RAG architecture:
  - **History-aware retriever** for better conversational memory,
  - **Retrieval QA chain** for smart document-based responses,
- Returns results with source documents and chat continuity.

### 💬 `main.py`

- Streamlit UI to interact with the bot,
- Sidebar user profile (name, email, avatar),
- Chat display with source references.

---

## 📌 Example Source

Currently ingested:

- [Herman Miller Chair Price Book (PB_CWB.pdf)](https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf)

Feel free to add more links to `documents_base_urls` in `ingestion.py`.

---

## 🛠 Future Enhancements

- Enable user-uploaded documents,
- Add login/authentication support,
- Improve error handling & feedback,
- Cloud deployment via Streamlit Cloud or Docker.

---

## 🔗 Tech Stack

- 🧠 [LangChain](https://www.langchain.com/)
- 🧠 [OpenAI](https://openai.com/)
- 📦 [Pinecone](https://www.pinecone.io/)
- 🌐 [Streamlit](https://streamlit.io/)
- 📦 [Pipenv](https://pipenv.pypa.io/)
