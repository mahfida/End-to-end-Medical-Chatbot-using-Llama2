from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.helper import download_embedding_model  # keep if it returns HuggingFaceEmbeddings


app = Flask(__name__)

# ✅ Load .env FIRST
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Check .env and load_dotenv().")

INDEX_NAME = "medical-chatbot"

# ✅ Embeddings must match what you used when upserting
embeddings = download_embedding_model()

# ✅ Connect to existing Pinecone index (Serverless)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ Create vectorstore wrapper (reuse existing vectors, no upsert)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# ✅ Prompt
prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQ: {question}\nA:")

# ✅ LLM (reduce max_new_tokens for speed + less context issues)
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 128, "temperature": 0.2},
)

# ✅ Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

def format_docs(docs, max_chars=600):
    return ("\n\n".join(d.page_content for d in docs))[:max_chars]

qa = (
    {"context": retriever | (lambda d: format_docs(d, 600)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def home():
    return render_template("chat.html")

# ✅ Add an endpoint to actually ask questions
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    answer = qa.invoke(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
