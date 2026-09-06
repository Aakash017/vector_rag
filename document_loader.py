"""
Document Loader - Reads PDF and TXT files
"""
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str):
    """Load a PDF file and return documents"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_text(file_path: str):
    """Load a text file and return documents"""
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks
