from langchain_chroma import Chroma


def create_vectorstore(chunks, embeddings, persist_directory="./chroma_db"):
    """
    Takes document chunks and embeddings model,
    creates a ChromaDB and stores the vectors.
    Returns the vectorstore object.
    """
    # Use Chroma.from_documents()
    # Parameters: documents, embedding, persist_directory
    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory)
    return vectorstore

def load_vectorstore(embeddings, persist_directory="./chroma_db"):
    """
    Loads an existing ChromaDB from disk.
    Returns the vectorstore object.
    """
    # Use Chroma() constructor
    # Parameters: persist_directory, embedding_function
    vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings)
    return vectorstore
