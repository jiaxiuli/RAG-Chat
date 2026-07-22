# 🧠 RAG Chat System (FastAPI + React + LangChain + LangGraph + Qdrant)

A full-stack AI-powered knowledge assistant that allows users to upload documents and ask questions based on their content.

RAG Chat uses Retrieval-Augmented Generation to find relevant information from uploaded documents and provide context-aware answers. The project combines modern AI frameworks, vector search, real-time response streaming, and full-stack application development.
It supports:

Uploading and parsing documents
Splitting documents into searchable text chunks
Generating embeddings using the OpenAI API
Storing and retrieving document chunks with Qdrant
Building RAG workflows with LangChain and LangGraph
Chatting with an LLM using uploaded documents as knowledge
Real-time streaming responses over WebSocket
Conversation history, token usage tracking, logging, and deletion APIs

---

## AI Workflow

LangChain provides reusable components for document processing, embeddings, retrieval, prompt management, and language-model integration.

LangGraph is used to organize the RAG process as a stateful workflow. It coordinates the main stages of query processing, retrieval, context preparation, answer generation, fallback handling, and result persistence.

The project uses a controlled RAG workflow rather than a fully autonomous AI agent.

---

### Preview
![RAGChat Screenshot](https://github.com/jiaxiuli/RAG-Chat/blob/main/images/RAGChat-3.png)

![RAGChat Screenshot](https://github.com/jiaxiuli/RAG-Chat/blob/main/images/RAGChat-1.png)

![RAGChat Screenshot - citation details](https://github.com/jiaxiuli/RAG-Chat/blob/main/images/RAGChat-2.png)

