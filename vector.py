import json
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

FAISS_INDEX_PATH = "faiss_index"
BM25_PATH = "bm25_retriever.pkl"


def load_documents():
    """Converts JSON datasets into LangChain Document objects."""
    docs = []

    # Index Patent Data
    if os.path.exists('patent.json'):
        with open('patent.json', 'r', encoding="utf-8") as f:
            patents = json.load(f)
            for p in patents:
                text = (
                    f"ID: {p['patent_id']}\n"
                    f"TITLE: {p['title']}\n"
                    f"SOLUTION: {p['proposed_solution']}\n"
                    f"LIMITS: {p['limitations']}"
                )
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"id": p['patent_id'], "type": "patent"}
                    )
                )

    # Index Gap Data
    if os.path.exists('patent_gap_dataset.json'):
        with open('patent_gap_dataset.json', 'r', encoding="utf-8") as f:
            gaps = json.load(f)
            for g in gaps:
                text = (
                    f"TARGET_ID: {g['patent_id']}\n"
                    f"GAP_TYPE: {g['gap_type']}\n"
                    f"REASON: {g['gap_reason']}\n"
                    f"RESEARCH: {g['potential_research_direction']}"
                )
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"id": g['patent_id'], "type": "gap"}
                    )
                )

    return docs


def build_db():
    """Builds and persists both FAISS and BM25 indices."""
    docs = load_documents()

    if not docs:
        print("⚠️ No documents found to index.")
        return

    print("📦 Building embeddings using Ollama...")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 1. Build FAISS
    print("🔹 Creating FAISS index...")
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(FAISS_INDEX_PATH)

    # 2. Build BM25
    print("🔹 Creating BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(docs)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"✅ Success: Optimized Hybrid Index created with {len(docs)} records.")


def get_retriever():
    """Loads indices and returns a weighted Hybrid Ensemble Retriever."""

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Load FAISS
    vector_db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    # Load BM25
    if os.path.exists(BM25_PATH):

        with open(BM25_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)

        bm25_retriever.k = 5

    else:
        docs = load_documents()
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5

    # Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3]
    )

    return ensemble_retriever


# ✅ MAIN EXECUTION BLOCK (THIS WAS MISSING)
if __name__ == "__main__":

    print("🚀 Starting Hybrid Vector DB setup...")

    # Step 1: Build database if not exists
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(BM25_PATH):

        print("📁 Index not found. Building new index...")
        build_db()

    else:
        print("📁 Existing index found. Skipping build.")

    # Step 2: Load retriever
    retriever = get_retriever()

    print("✅ Hybrid Retriever Ready!")