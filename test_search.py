"""
Clean test to verify vectorstore search functionality
"""
# Suppress embedding debug output
import warnings
warnings.filterwarnings('ignore')

from src.rag.embeddings import model
from src.rag.vectorstore import load_vectorstore

# Load existing vectorstore
vectorstore = load_vectorstore(model, "vectorstore")

# Test similarity search
query = "What is this document about?"
print(f"\n{'='*60}")
print(f"Query: '{query}'")
print(f"{'='*60}")

results = vectorstore.similarity_search(query, k=2)

print(f"\nFound {len(results)} results:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 60)

# Test with another query
query2 = "Tell me about power plants"
print(f"\n{'='*60}")
print(f"Query: '{query2}'")
print(f"{'='*60}")

results2 = vectorstore.similarity_search(query2, k=2)

print(f"\nFound {len(results2)} results:\n")

for i, doc in enumerate(results2, 1):
    print(f"Result {i}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 60)
