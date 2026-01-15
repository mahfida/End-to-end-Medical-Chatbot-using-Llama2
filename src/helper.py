
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings





# Load PDF files from a directory
def load_pdf(data):
    loader=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs


# Create text splitter
def text_splitter(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        #separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download embedding model and create embeddings 
def download_embedding_model():  
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model