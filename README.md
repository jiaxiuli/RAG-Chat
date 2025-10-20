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

### Preview
![RAGChat Screenshot](https://github.com/jiaxiuli/RAG-Chat/blob/main/images/RAGChat-1.png)
![RAGChat Screenshot - citation details](https://github.com/jiaxiuli/RAG-Chat/blob/main/images/RAGChat-2.png)


### Environment Requirements

- Python ≥ 3.11
- PostgreSQL ≥ 15  
- [pgvector](https://github.com/pgvector/pgvector) extension installed  
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- 
- Before running the project, make sure your environment meets the following requirements:

PostgreSQL: Installed and configured locally
Node.js & npm: Installed (for the front-end)

Set up your local PostgreSQL database using the following connection string:
DATABASE_URL = "postgresql://rag_user:123456@localhost:5432/ragchat"

---

### 🔑 Step 1. Clone this repo

```bash
git clone https://github.com/yourusername/RAGChat.git

```
### Step 2. Backend Setup
Navigate to the Back-end folder:
```bash
cd Back-end
```

Create a .env file inside the Back-end folder and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Step 3. Frontend Setup
Navigate to the Front-end folder:
```bash
cd Front-end
```

Install npm dependencies:
```bash
npm install
```

### Step 4. Running the Project
##### Backend Server

From the Back-end/app directory, run:
```bash
python main.py
```
##### Frontend

From the Front-end directory, start the React app:
```bash
npm start
```
