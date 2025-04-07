# Herman Miller Product Helper Bot 🪑🤖

An AI-powered chatbot that helps users query detailed product information, pricing breakdowns, finishes, features, and illustrations from Herman Miller’s pricing documents. Powered by **LangChain**, **OpenAI**, **Pinecone**, and **Streamlit**.

---

## 📦 Features

- Extracts part numbers, configurations, and feature blocks (e.g., Surface Materials, MicrobeCare).
- Parses complex price tables including finish variations (e.g., Metallic Paint).
- Uses OCR for image-only pages.
- Displays illustrations linked to product parts.
- Supports timestamped chat history export with Gravatar profile integration.

---

## 📁 Project Structure

| File         | Description |
|--------------|-------------|
| `ingestion.py` | Ingests product data from PDFs with OCR, pricing, and image extraction. |
| `core.py`      | Backend RAG logic: retrieval, prompt formatting, and LLM invocation. |
| `main.py`      | Streamlit UI for chatting with the bot and exporting chat history. |
| `Pipfile`/`Pipfile.lock` | Dependency management via Pipenv. |

---

## 🚀 Getting Started

### 1. Install Dependencies (Pipenv)

```bash
pip install pipenv
pipenv install
pipenv shell
```

---

### 2. Environment Setup

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
INDEX_NAME=hermanmiller-product-helper
```

---

### 3. Ingest Product PDF(s)

Run this to download, parse, and upload the product data:

```bash
python ingestion.py
```

It will:

- Download the Herman Miller PDF
- Use OCR for non-text pages
- Extract images and price tables
- Embed content with `text-embedding-3-large`
- Store it in your Pinecone index

---

### 4. Run the Chatbot App

```bash
streamlit run main.py
```

Then you’ll be able to:

- Ask product questions (e.g., “Show me FT123 pricing and images”)
- View images and markdown pricing tables
- Export chat history with timestamps

---

## 🧠 How It Works

### 🔍 `ingestion.py`
- Downloads and parses pricing PDFs
- Uses OCR when text isn’t directly extractable
- Detects part numbers like `FT123`, extracts features and images
- Extracts pricing with support for finish upcharges

### 🤖 `core.py`
- Classifies query intent (e.g., pricing, image, feature)
- Retrieves relevant chunks using filters (like part number or feature block)
- Passes context to GPT via a structured prompt

### 💬 `main.py`
- Provides a modern Streamlit interface with:
  - Real-time Q&A
  - Image display
  - User info (avatar/name)
  - JSON export of full chat history with timestamps

---

## 🧪 Example Supported Queries

- “What’s the price of FT165 with metallic finishes?”
- “I'm considering some Herman Miller products can you tell me the price for a 6" screen for FV694?"
- “What surface materials are available?”
- “Give me MicrobeCare options”
- “Show me images of part FT123”
---

## 📎 Data Source

- [PB_CWB.pdf – Canvas Office Landscape Wall & Private Office Price Book](https://www.hermanmiller.com/content/dam/hermanmiller/documents/pricing/PB_CWB.pdf)

---

## 🔧 Future Plans

- Add multi-PDF ingestion and organization
- Cloud deployment (Streamlit Cloud, Docker)

---

## 🔗 Tech Stack

- 🧠 [LangChain](https://www.langchain.com/)
- ✨ [OpenAI GPT-4 Turbo](https://platform.openai.com/)
- 🧱 [Pinecone Vector DB](https://www.pinecone.io/)
- 🌐 [Streamlit](https://streamlit.io/)
- 📦 [Pipenv](https://pipenv.pypa.io/)
