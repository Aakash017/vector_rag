# Vector RAG System - Interview Preparation Guide

## Project Overview

**Project Name:** Vector RAG (Retrieval-Augmented Generation)  
**Purpose:** A document processing and semantic search system that uses vector embeddings to enable intelligent information retrieval  
**Tech Stack:** Python, LangChain, AWS Bedrock, ChromaDB, UV package manager

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines:
1. **Information Retrieval:** Finding relevant documents from a knowledge base
2. **Generation:** Using an LLM to generate answers based on retrieved context

**Why RAG?**
- Reduces hallucinations by grounding LLM responses in actual documents
- Enables LLMs to access up-to-date or private information
- More cost-effective than fine-tuning for domain-specific knowledge

---

## System Architecture

```
Document Source (PDF/TXT)
         ↓
   Document Loader
         ↓
    Text Splitter (Chunks)
         ↓
  Embedding Model (AWS Bedrock)
         ↓
  Vector Database (ChromaDB)
         ↓
   Similarity Search
         ↓
  Relevant Context Retrieved
```

---

## Key Components Explained

### 1. Document Loaders (`document_loader.py`)

**Purpose:** Load and preprocess documents from various sources

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_text(file_path: str):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents
```

**Key Points:**
- Supports multiple formats (PDF, TXT)
- Returns LangChain Document objects with content and metadata
- Handles text extraction from different file types

### 2. Text Splitting

**Purpose:** Break large documents into smaller, manageable chunks

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks
```

**Why Split Documents?**
- Embedding models have token limits
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost at chunk boundaries

**Configuration:**
- `chunk_size=500`: Each chunk ~500 characters
- `chunk_overlap=20`: 20-character overlap between chunks

### 3. Embeddings (`src/rag/embeddings.py`)

**Purpose:** Convert text into numerical vector representations

```python
from langchain_aws import BedrockEmbeddings

model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)
```

**What are Embeddings?**
- High-dimensional vectors (arrays of numbers) that represent semantic meaning
- Similar concepts have similar vectors (measured by cosine similarity)
- Enable semantic search (meaning-based, not keyword-based)

**Example:**
- "hello" vs "hi there": similarity = 0.634 (high, similar meaning)
- "hello" vs "pizza": similarity = 0.269 (low, different meaning)

**Why AWS Bedrock?**
- Managed service (no model hosting required)
- Access to state-of-the-art embedding models
- Scalable and serverless

### 4. Vector Store (`src/rag/vectorstore.py`)

**Purpose:** Store and retrieve document embeddings efficiently

```python
from langchain_chroma import Chroma

def create_vectorstore(chunks, embeddings, persist_directory="./chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def load_vectorstore(embeddings, persist_directory="./chroma_db"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore
```

**Why ChromaDB?**
- Open-source vector database
- Easy to set up (no separate server required)
- Persistent storage (saved to disk)
- Fast similarity search using ANN (Approximate Nearest Neighbors)

**Key Operations:**
- `create_vectorstore`: Embeds documents and stores vectors
- `load_vectorstore`: Loads existing vector database
- `similarity_search`: Finds most relevant chunks for a query

### 5. Similarity Search

**How It Works:**
1. Query text is converted to an embedding vector
2. Vector database compares query vector with all stored vectors
3. Returns top-k most similar documents (by cosine similarity)

```python
results = vectorstore.similarity_search(query, k=2)
```

**Parameters:**
- `query`: The search question/text
- `k`: Number of results to return

---

## Project Workflow

### Setup Phase
1. Install dependencies with `uv sync`
2. Configure AWS credentials in `.env`
3. Activate virtual environment

### Data Ingestion Phase
1. Load documents using appropriate loader
2. Split documents into chunks
3. Generate embeddings for each chunk
4. Store embeddings in ChromaDB

### Query Phase
1. User submits a query
2. Query is embedded using the same model
3. Vector similarity search finds relevant chunks
4. Results are returned with metadata

---

## Technical Concepts for Interviews

### Vector Similarity

**Cosine Similarity Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```
- Range: -1 to 1 (higher = more similar)
- Used to measure semantic similarity between embeddings

### Embedding Dimensions
- Amazon Titan embeddings: 1,536 dimensions
- Each dimension captures different semantic features
- High-dimensional space allows nuanced meaning representation

### Chunking Strategy
- **Trade-off:** Smaller chunks = precise but less context; Larger chunks = more context but less precise
- **Overlap:** Prevents losing context at boundaries
- **Recursive splitter:** Tries to split on natural boundaries (paragraphs, sentences, words)

---

## Sample Interview Questions & Answers

### Q1: "Explain how your RAG system works"

**Answer:**
"I built a RAG system that processes documents and enables semantic search. First, I load documents using LangChain loaders, then split them into 500-character chunks with 20-character overlap to balance precision and context. Each chunk is embedded using AWS Bedrock's Titan model, which converts text into 1,536-dimensional vectors capturing semantic meaning. These embeddings are stored in ChromaDB, a vector database. When a user queries, the system embeds the query, performs cosine similarity search against stored vectors, and retrieves the most relevant chunks. This enables intelligent, meaning-based search rather than simple keyword matching."

### Q2: "Why did you choose ChromaDB over other vector databases?"

**Answer:**
"I chose ChromaDB because it's lightweight and doesn't require a separate server, making it perfect for development and prototyping. It persists data to disk, supports filtering, and has excellent integration with LangChain. For production, I'd consider alternatives like Pinecone for managed cloud solutions, or Weaviate for more advanced features, depending on scale and requirements."

### Q3: "What's the difference between semantic search and keyword search?"

**Answer:**
"Keyword search matches exact words, so 'automobile' won't match 'car'. Semantic search understands meaning - it converts text to embeddings, so 'automobile' and 'car' have similar vectors and will match. This is powerful because users can search using natural questions, and the system understands intent rather than just matching words. For example, searching 'tell me about power plants' returns content about power facilities even if those exact words don't appear."

### Q4: "How do you handle documents that are too large for embedding models?"

**Answer:**
"I use text splitting with RecursiveCharacterTextSplitter. It breaks documents into chunks of 500 characters with 20-character overlap. The overlap prevents losing context at boundaries. The recursive approach tries to split on natural boundaries like paragraphs and sentences first, preserving semantic coherence. Each chunk is embedded and stored separately, and during retrieval, the most relevant chunks are returned."

### Q5: "What are embeddings and why are they useful?"

**Answer:**
"Embeddings are dense vector representations of text in high-dimensional space, where semantically similar concepts are positioned close together. They're useful because they capture meaning mathematically, enabling operations like similarity comparison using cosine distance. In my system, I use Amazon Titan embeddings which produce 1,536-dimensional vectors. These embeddings power semantic search, allowing the system to understand that 'power plant' and 'energy facility' are related concepts."

### Q6: "How would you scale this system for production?"

**Answer:**
"Several approaches:
1. **Database:** Move from ChromaDB to a managed vector database like Pinecone or use PostgreSQL with pgvector extension
2. **Caching:** Cache frequently queried embeddings to reduce API calls
3. **Batch Processing:** Embed documents in batches asynchronously
4. **API Layer:** Wrap the system in a FastAPI service with rate limiting
5. **Monitoring:** Add logging for query latency, embedding costs, and result quality metrics
6. **Indexing:** Use approximate nearest neighbor algorithms (HNSW, IVF) for faster search at scale"

### Q7: "What challenges did you face and how did you solve them?"

**Answer:**
"The main challenges were:
1. **Chunking Strategy:** Finding the right chunk size that balances context and precision. I settled on 500 characters with 20-character overlap after testing.
2. **Environment Setup:** Configured AWS credentials securely using .env files and ensured they're in .gitignore
3. **Dependency Management:** Used UV package manager for faster, more reliable Python dependency resolution
4. **Testing:** Verified embeddings work correctly by testing cosine similarity between known similar/dissimilar phrases"

### Q8: "How do you measure the quality of your search results?"

**Answer:**
"Several approaches:
1. **Relevance Testing:** Manually verify top-k results make sense for test queries
2. **Similarity Scores:** Check cosine similarity scores - higher scores (closer to 1) indicate better matches
3. **Metrics:** Use standard IR metrics like Precision@K, Recall@K, and MRR (Mean Reciprocal Rank)
4. **A/B Testing:** Compare different chunking strategies or embedding models
5. **User Feedback:** In production, collect implicit feedback (clicks, dwell time) and explicit feedback (thumbs up/down)"

---

## Technical Details

### Dependencies
```toml
[project]
dependencies = [
    "boto3>=1.43.70",              # AWS SDK
    "chromadb>=1.5.9",             # Vector database
    "langchain>=1.4.0",            # LLM framework
    "langchain-aws>=1.7.5",        # AWS integrations
    "langchain-chroma>=1.1.0",     # ChromaDB integration
    "langchain-community>=0.4.2",  # Community loaders
    "langchain-text-splitters>=0.4.0",  # Text splitting
    "pypdf>=6.17.0",               # PDF processing
]
```

### Environment Variables (`.env`)
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

### Project Structure
```
vector_rag/
├── src/
│   └── rag/
│       ├── embeddings.py      # Embedding model configuration
│       ├── vectorstore.py     # Vector database operations
│       └── __init__.py
├── document_loader.py         # Document loading utilities
├── test_loader.py            # Basic tests
├── test_search.py            # Search functionality tests
├── create_vectorstore.py     # Vectorstore creation script
├── sample.txt                # Sample document
├── vectorstore/              # Persisted ChromaDB data (gitignored)
├── .env                      # AWS credentials (gitignored)
├── .venv/                    # Virtual environment (gitignored)
├── pyproject.toml            # Project dependencies
└── README.md                 # Project documentation
```

---

## Code Snippets to Remember

### Complete RAG Pipeline Example

```python
# 1. Load documents
from document_loader import load_text, split_documents

docs = load_text("sample.txt")
chunks = split_documents(docs, chunk_size=500, chunk_overlap=20)

# 2. Create embeddings and vectorstore
from src.rag.embeddings import model
from src.rag.vectorstore import create_vectorstore

vectorstore = create_vectorstore(chunks, model, "vectorstore")

# 3. Query the system
query = "What is this document about?"
results = vectorstore.similarity_search(query, k=2)

for doc in results:
    print(doc.page_content)
```

---

## Best Practices Demonstrated

1. **Configuration Management:** Used `.env` for sensitive credentials
2. **Dependency Management:** Used `pyproject.toml` and UV for reproducible builds
3. **Modular Design:** Separated concerns (loading, embedding, storage)
4. **Git Hygiene:** Proper `.gitignore` (excluded .env, .venv, .kiro, vectorstore)
5. **Testing:** Created test scripts to verify functionality
6. **Documentation:** Clear README and code comments

---

## Performance Considerations

### Costs
- **AWS Bedrock Titan Embeddings:** ~$0.0001 per 1K tokens
- **Storage:** ChromaDB is free (local SQLite)

### Latency
- **Embedding Generation:** ~100-300ms per chunk
- **Similarity Search:** <50ms for small datasets (<10K documents)

### Optimization Tips
1. Batch embed documents to reduce API calls
2. Cache embeddings for frequently accessed documents
3. Use approximate nearest neighbor search for large datasets
4. Consider dimensionality reduction for very large scale

---

## Future Enhancements

1. **Hybrid Search:** Combine semantic search with keyword search (BM25)
2. **Reranking:** Add a reranking model to improve result quality
3. **Multi-modal:** Support images, tables, and other content types
4. **Query Expansion:** Automatically expand queries for better recall
5. **Metadata Filtering:** Filter results by document type, date, etc.
6. **API Interface:** Build a FastAPI endpoint for programmatic access
7. **Evaluation Suite:** Add automated testing with ground truth datasets

---

## Common Pitfalls & Solutions

### Problem: Poor search results
**Solution:** Adjust chunk size, try different overlap values, or use a better embedding model

### Problem: Slow embedding generation
**Solution:** Batch process documents, use async API calls, or consider local embedding models

### Problem: AWS credential errors
**Solution:** Verify `.env` file is loaded, check IAM permissions for Bedrock access

### Problem: Empty search results
**Solution:** Ensure vectorstore was created before loading, verify documents were actually embedded

---

## Key Takeaways

✅ **RAG = Retrieval + Generation** (this project focuses on the retrieval part)  
✅ **Embeddings capture semantic meaning** as vectors  
✅ **Vector databases enable fast similarity search**  
✅ **Chunking strategy impacts search quality**  
✅ **Production systems need caching, monitoring, and scalability considerations**

---

## Demo Script for Interviews

```bash
# 1. Show project structure
ls -la

# 2. Show dependencies
cat pyproject.toml

# 3. Create vectorstore
uv run python create_vectorstore.py

# 4. Run search test
uv run python test_search.py

# 5. Explain results
# - Show how relevant chunks are retrieved
# - Discuss cosine similarity scores
# - Explain why certain results rank higher
```

---

## Additional Resources

- **LangChain Docs:** https://python.langchain.com/docs/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **AWS Bedrock:** https://aws.amazon.com/bedrock/
- **Vector Embeddings Guide:** https://www.pinecone.io/learn/vector-embeddings/

---

**Last Updated:** October 2026  
**Author:** Aakash (Aakash017)  
**Project:** Vector RAG System
