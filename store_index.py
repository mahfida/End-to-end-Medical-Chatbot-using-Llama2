from src.helper import load_pdf, text_splitter, download_embedding_model
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()  # loads .env from current directory or parents
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print("Key loaded:", PINECONE_API_KEY is not None)



# Load PDF documents from directory
extracted_data = load_pdf('data/')
print(f"✅ {len(extracted_data)} documents loaded")

# Split documents into text chunks
text_chunks = text_splitter(extracted_data)
print(f"✅ {len(text_chunks)} text chunks created")

# Download embedding model
embeddings = download_embedding_model()
print("✅ Embedding model loaded")  

# Create Pinecone Index and Vector Store if not existing
INDEX_NAME = "medical-chatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
# Flag to control upsert
index_created = False
# Create index ONLY if it does not exist
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # must match embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    index_created = True
    print("✅ Index created")
else:
    print("ℹ️ Index already exists — reusing it")
# Connect to the index (always)
index = pc.Index(INDEX_NAME)
# Create LangChain vector store wrapper
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
# Upsert documents ONLY if index was just created
if index_created:
    vectorstore.add_documents(text_chunks)
    print("✅ Documents upserted")
else:
    print("🔎 Skipping upsert (existing vectors reused)")

