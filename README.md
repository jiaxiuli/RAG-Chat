# 🧠 RAG Chat System (FastAPI + React + PostgreSQL + pgvector)

This project is a **Retrieval-Augmented Generation (RAG)** chat system built with **FastAPI** (backend), **React + Zustand** (frontend), and **PostgreSQL with pgvector** for embedding storage.

It supports:
- Uploading and parsing PDFs  
- Generating embeddings using the OpenAI API  
- Storing and searching document chunks via pgvector  
- Chatting with an LLM using your own documents as knowledge  
- Real-time streaming responses over WebSocket  
- Conversation history, logging, and deletion APIs  

---

## 🚀 Features

| Feature | Description |
|----------|-------------|
| **PDF Upload** | Parse PDF, chunk into text blocks, embed, and store in database. |
| **Hybrid Search** | Combine semantic (pgvector) and keyword (BM25) retrieval. |
| **Chat Interface** | Ask questions with context-aware LLM responses (OpenAI GPT model). |
| **Streaming Responses** | Real-time streaming via WebSocket. |
| **Conversation Memory** | All chats are saved and can be retrieved or deleted later. |
| **Structured Backend** | Built with FastAPI and SQLAlchemy ORM. |

---


---

## ⚙️ 1. Backend Setup

### 🐍 Prerequisites

- Python ≥ 3.10  
- PostgreSQL ≥ 15  
- [pgvector](https://github.com/pgvector/pgvector) extension installed  
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

---

### 🔑 Step 1. Clone this repo

```bash
git clone https://github.com/yourusername/RAGChat.git
cd RAGChat/Back-end/app
```
### Step 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

### Step 3. Install dependencies
If you already have a requirements.txt:

pip install -r requirements.txt

fastapi
uvicorn
sqlalchemy
psycopg2-binary
openai
tiktoken
python-dotenv
PyPDF2
requests

### Step 4. Install and configure PostgreSQL + pgvector
macOS
brew install postgresql
brew services start postgresql
psql postgres
CREATE DATABASE ragchat;
\c ragchat
CREATE EXTENSION vector;

Ubuntu / Linux
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql
CREATE DATABASE ragchat;
\c ragchat
CREATE EXTENSION vector;

Windows (WSL or local)
Install PostgreSQL from postgresql.org/download
.

After installation, open psql and run:

CREATE DATABASE ragchat;
\c ragchat
CREATE EXTENSION vector;

### Step 5. Create the .env file
Inside the Back-end/app/ directory, create a new file named .env:

# .env
DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/ragchat
OPENAI_API_KEY=sk-your-openai-key

### Step 6. Create the database tables
python -m main

### Step 7. Run the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Backend is now running at:
👉 http://localhost:8000

Interactive API docs:
👉 http://localhost:8000/docs


