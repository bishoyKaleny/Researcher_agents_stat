# agents/retriever.py

from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import json

class LocalSentenceTransformers:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, batch_size=16, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0]

    def __call__(self, text):
        return self.embed_query(text)

def load_retriever():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = LocalSentenceTransformers(model)
    vectorstore = FAISS.load_local("rebuilt_faiss_langchain", embeddings, allow_dangerous_deserialization=True)

    def search_and_serialize(query):
        docs = vectorstore.similarity_search(query, k=20)
        filtered_docs = [
            doc for doc in docs if doc.metadata.get("type") == "NarrativeText"
        ]
        return json.dumps([
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in filtered_docs
        ])

    return Tool(
        name="retrieve_context",
        func=search_and_serialize,
        description="Retrieve only NarrativeText-type documents relevant to the query as a JSON string."
    )

