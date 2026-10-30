"""
Create and populate the vectorstore with sample documents
"""
import warnings
warnings.filterwarnings('ignore')

from document_loader import load_text, split_documents
from src.rag.embeddings import model
from src.rag.vectorstore import create_vectorstore

print("Loading document...")
docs = load_text("sample.txt")
print(f"[OK] Loaded {len(docs)} document(s)")

print("\nSplitting into chunks...")
chunks = split_documents(docs, chunk_size=500, chunk_overlap=20)
print(f"[OK] Created {len(chunks)} chunks")

print("\nChunk contents:")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk.page_content)
    print("-" * 60)

print("\nCreating vectorstore...")
vectorstore = create_vectorstore(chunks, model, "vectorstore")
print("[OK] Vectorstore created successfully!")

print(f"\nTotal documents in vectorstore: {vectorstore._collection.count()}")
