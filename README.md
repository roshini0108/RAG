# RAG Customer Support Assistant

This project is a beginner-friendly Retrieval-Augmented Generation (RAG) customer support assistant built with:

- Python
- LangChain
- LangGraph
- ChromaDB
- PyPDFLoader
- Ollama

## Project Structure

```text
RAG/
|-- main.py
|-- ingestion.py
|-- retrieval.py
|-- graph.py
|-- hitl.py
|-- config.py
|-- requirements.txt
|-- .env.example
|-- README.md
`-- data/
```

## What the Project Does

1. Loads PDF documents from the `data/` folder
2. Splits them into chunks
3. Creates embeddings with Ollama
4. Stores them in ChromaDB
5. Retrieves the top matching chunks
6. Uses an Ollama LLM to answer support questions
7. Uses LangGraph to manage workflow steps
8. Escalates to human review when confidence is low or the query is complex

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install and prepare Ollama

Install Ollama locally, then pull the chat and embedding models:

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and adjust the model names if needed.

### 5. Add PDFs

Place your support PDFs inside the `data` folder.

### 6. Run the assistant

```powershell
python main.py
```

## Optional: Run ingestion manually

```powershell
python ingestion.py
```

## Example Flow

- User asks a support question
- System retrieves top matching chunks from ChromaDB
- If confidence is good, Ollama generates the answer
- If confidence is low, no documents are found, or the query is complex, the system asks for human input

## Error Handling Included

- Missing PDF files
- Missing Chroma database
- No retrieval matches
- Empty query
- Ollama generation failures
