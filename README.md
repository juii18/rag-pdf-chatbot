# 📄 RAG PDF Chatbot

An AI-powered PDF chatbot that allows users to upload a PDF and ask questions about its content using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Intelligent document chunking
- 🧠 HuggingFace sentence embeddings
- 🔎 Semantic similarity search
- 🗄️ ChromaDB vector database
- 🤖 Groq LLM for answer generation
- 💬 Ask questions using natural language
- 📚 Answers grounded in the uploaded PDF
- 🔐 Environment-based API key configuration

## 🏗️ RAG Architecture

PDF Upload  
↓  
PDF Document Loader  
↓  
Text Chunking  
↓  
HuggingFace Embeddings  
↓  
ChromaDB Vector Store  
↓  
Similarity Search  
↓  
Relevant Context  
↓  
Groq LLM  
↓  
AI Generated Answer

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Groq
- Llama 3.1
- PyPDF

## 📁 Project Structure

```text
rag-pdf-chatbot/
│
├── data/
│   └── sample.pdf
│
├── src/
│   ├── app.py
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag.py
│
├── .gitignore
├── requirements.txt
└── README.md
