from document_loader import load_text, split_documents
from src.rag.embeddings import model



# # Load the document
# docs = load_text("sample.txt")
# print(f"Loaded {len(docs)} document(s)")
# print(f"Content preview: {docs[0].page_content[:500]}...")

# # Split into chunks
# chunks = split_documents(docs, chunk_size=500, chunk_overlap=20)
# print(f"\nSplit into {len(chunks)} chunks")

# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:")
#     print(chunk.page_content)
#     print("-" * 50)


# # Test Vector based store and loading

from src.rag.vectorstore import load_vectorstore, create_vectorstore

# # Load the document
# docs = load_text("sample.txt")
# print(f"Loaded {len(docs)} document(s)")
# print(f"Content preview: {docs[0].page_content[:500]}...")

# # Split into chunks
# chunks = split_documents(docs, chunk_size=500, chunk_overlap=20)
# print(f"\nSplit into {len(chunks)} chunks")

# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:")
#     print(chunk.page_content)
#     print("-" * 50)

# # Create vectorstore
# vectorstore = create_vectorstore(chunks, model, "vectorstore")
# print(vectorstore, "completed..")
vectorstore = load_vectorstore(model, "vectorstore")


# Test similarity search
query = "What is this document about?"
results = vectorstore.similarity_search(query, k=2)

print(f"\n\nSearch results for: '{query}'")
print("=" * 50)
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print("-" * 50)
