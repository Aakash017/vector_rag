# Vector RAG

A step-by-step learning project for building a Retrieval-Augmented Generation (RAG) system using vector embeddings.

## Tech Stack

- **Embeddings**: AWS Bedrock Titan (`amazon.titan-embed-text-v2:0`)
- **Vector Store**: ChromaDB
- **LLM**: AWS Bedrock Claude
- **Framework**: LangChain

## Project Structure

```
vector-rag/
├── src/rag/
│   ├── __init__.py
│   ├── embeddings.py     # Bedrock Titan embeddings
│   └── vectorstore.py    # ChromaDB operations
├── document_loader.py    # PDF and text file loading + chunking
├── test_loader.py        # Testing and experimentation
├── sample.txt            # Sample document for testing
└── vectorstore/          # ChromaDB persistent storage
```

## Concepts Learned

### RAG Pipeline
```
Document → Chunks → Embeddings → Vector DB → Search → LLM → Answer
```

### Key Concepts

1. **Chunking**: Split documents into smaller pieces
   - Smaller chunks = more precise retrieval
   - Larger chunks = more context per result

2. **Embeddings**: Convert text to vectors (1024 dimensions with Titan v2)
   - Similar meanings = similar vectors
   - Enables semantic search (meaning-based, not keyword-based)

3. **Vector Store**: Database optimized for similarity search
   - Stores text + embedding vectors
   - Finds closest vectors to a query

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure AWS credentials (`.env` file):
   ```
   AWS_ACCESS_KEY_ID=your_key
   AWS_SECRET_ACCESS_KEY=your_secret
   AWS_DEFAULT_REGION=us-east-1
   ```

3. Run tests:
   ```bash
   python test_loader.py
   ```

## Progress

- [x] Document loading (PDF, TXT)
- [x] Text chunking with overlap
- [x] Embeddings with Bedrock Titan
- [x] Vector storage with ChromaDB
- [x] Similarity search
- [ ] RAG chain with Claude
- [ ] Query and answer generation

## Next Steps

- Build the RAG chain connecting retrieval to Claude
- Implement question-answering over documents

