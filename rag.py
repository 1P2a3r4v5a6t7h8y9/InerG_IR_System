import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# config - update if needed
LOCAL_MODEL_PATH = r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\models\bge-small-en-v1.5"
VECTOR_STORE_DIR = r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\IR_SYSTEM\Scripts\vector_store"  # adjust if your vector_store path differs
EMBED_DIM = None  # will be inferred after loading model

# load .env for optional Gemini use
load_dotenv()

# load local embedding model once (re-used)
print("Loading local embedding model for queries...")
embed_model = SentenceTransformer(LOCAL_MODEL_PATH)


def load_faiss_and_meta(vector_store_dir=VECTOR_STORE_DIR):
    """Load FAISS index and metadata JSON."""
    idx_path = os.path.join(vector_store_dir, "faiss.index")
    meta_path = os.path.join(vector_store_dir, "meta.json")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"FAISS index or meta not found in {vector_store_dir}")

    index = faiss.read_index(idx_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def embed_query(text):
    """Return 1-D numpy vector (normalized) for the query using local model."""
    v = embed_model.encode([text], normalize_embeddings=True)
    return np.array(v[0], dtype="float32")


def search_faiss(index, query_vec, top_k=10):
    """Search FAISS index; returns (indices, distances). Distances are L2; convert to cosine-like via dot if normalized."""
    # If index is L2, and embeddings are normalized, smaller L2 -> larger cosine.
    query_vec = np.array(query_vec).reshape(1, -1).astype("float32")
    D, I = index.search(query_vec, top_k)
    return I[0].tolist(), D[0].tolist()


def rank_results_by_cosine(query_vec, hits_indices, metadata, index_vectors=None):
    """
    Compute cosine similarity between query_vec and retrieved metadata texts.
    If index_vectors provided (numpy array of stored vectors), compute dot product; otherwise use distances returned by FAISS.
    Returns list of dicts: [{'score':..., 'text':..., 'file_name':...}, ...]
    """
    results = []

    # Prefer direct dot similarity if index_vectors supplied
    if index_vectors is not None:
        # index_vectors shape: (N, dim)
        for idx in hits_indices:
            v = index_vectors[idx]
            score = float(np.dot(query_vec, v))  # both should be normalized
            m = metadata[idx]
            results.append({"score": score, "text": m.get("text", "")[:400], "file_name": m.get("file_name", "")})
    else:
        # Fallback: we will load the textual metadata only and compute scores using local embedding (slower)
        for idx in hits_indices:
            m = metadata[idx]
            text = m.get("text", "")
            emb = embed_model.encode([text], normalize_embeddings=True)[0]
            score = float(np.dot(query_vec, emb))
            results.append({"score": score, "text": text[:400], "file_name": m.get("file_name", "")})

    # sort descending by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


def get_index_vectors(index):
    """
    If the FAISS index is IndexFlatL2 containing raw vectors, try to extract the stored vectors as numpy array.
    Note: this reads index.reconstruct for each id which can be slow for huge DBs.
    """
    try:
        ntotal = index.ntotal
        dim = index.d
        arr = np.zeros((ntotal, dim), dtype="float32")
        for i in range(ntotal):
            arr[i] = index.reconstruct(i)
        return arr
    except Exception:
        return None
