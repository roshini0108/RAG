# 🚀 RAG-Based Customer Support Assistant

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for customer support.
It processes a knowledge base (PDF/documents), retrieves relevant information using embeddings, and generates accurate, context-aware responses using an LLM.

The system is designed with **LangGraph workflow orchestration** and supports **Human-in-the-Loop (HITL)** escalation for low-confidence or sensitive queries.

---

## 🎯 Key Features

* 📄 **Document-based Q&A** using RAG
* 🔍 **Semantic Retrieval** with ChromaDB (HNSW index)
* 🧠 **Embeddings** using OpenAI `text-embedding-3-small`
* ⚡ **MMR + Cross-Encoder Reranking** for better accuracy
* 🔁 **LangGraph Workflow** for structured execution
* 🎯 **Confidence-based Routing**

  * Auto Answer
  * Clarification
  * HITL Escalation
* 👨‍💻 **Human-in-the-Loop (HITL)** support for complex queries

---

## 🏗️ System Architecture

```
User Query
    ↓
Query Processing (Intent + Embedding)
    ↓
Retriever (ChromaDB + MMR)
    ↓
Reranker (Cross-Encoder)
    ↓
Router (Confidence-based)
   ↙        ↓        ↘
Auto     Clarify     HITL
   ↓
LLM (GPT-4o-mini)
   ↓
Final Response
```

---

## 🔄 Workflow

1. **Document Ingestion**

   * Load PDF → Chunk (800 tokens, 120 overlap)
   * Generate embeddings
   * Store in ChromaDB

2. **Query Processing**

   * User query → embedding
   * Retrieve top chunks using MMR
   * Rerank using cross-encoder

3. **Response Generation**

   * Pass context to LLM
   * Generate grounded answer

4. **Routing**

   * Confidence ≥ 0.72 → Auto Answer
   * Ambiguous → Clarification
   * Low confidence / sensitive → HITL

---

## 🧠 Tech Stack

* **Backend:** Python
* **Vector DB:** ChromaDB
* **Embeddings:** OpenAI `text-embedding-3-small`
* **LLM:** GPT-4o-mini (fallback: Claude Haiku)
* **Workflow Engine:** LangGraph
* **Reranking:** Cross-Encoder (MiniLM)

---

## 📂 Project Structure

```
RAG-based-customer-support/
│
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── graph.py
│   ├── hitl.py
│   └── main.py
│
├── docs/
│   ├── HLD.pdf
│   ├── LLD.pdf
│   └── TechnicalDoc.pdf
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Clone the repo

```
git clone https://github.com/roshini0108/RAG-based-customer-support.git
cd RAG-based-customer-support
```

### 2. Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the project

```
python main.py
```

---

## 📊 Routing Logic

| Condition                  | Action      |
| -------------------------- | ----------- |
| High confidence (≥ 0.72)   | Auto Answer |
| Medium / ambiguous         | Clarify     |
| Low confidence / sensitive | HITL        |

---

## 🧪 Example Queries

* “How do I reset my password?”
* “What is your refund policy?”
* “I was charged twice, what should I do?”

---

## 🔮 Future Enhancements

* Multi-document support
* Feedback-based learning
* Conversational memory
* Deployment with FastAPI
* Scalable cloud architecture

---

## 🏆 Key Highlights

* Combines **retrieval + generation + decision logic**
* Uses **graph-based workflow (LangGraph)**
* Ensures reliability with **HITL escalation**
* Designed for **real-world scalability**

---

## 👩‍💻 Author

Mutyala Roshini

---

## 📜 License

This project is for academic and learning purposes.
