# import os
# import json
# import faiss
# import numpy as np
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load API Key
# load_dotenv(r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\IR_SYSTEM\Scripts\.env")
# api_key = os.getenv("GOOGLE_API_KEY")
# print("Loaded API key:", "FOUND" if api_key else "NOT FOUND")

# genai.configure(api_key=api_key)

# EMBED_MODEL = "models/text-embedding-004"
# BATCH_SIZE = 20   # 🔥 SAFER for Gemini


# # ----------------------------------------------------------
# # 1. SAFE EMBEDDING FUNCTION
# # ----------------------------------------------------------
# def embed_text_batch(text_list):
#     """Embeds text one-by-one using Gemini (batch not supported)."""
#     embeddings = []

#     for i, text in enumerate(text_list):
#         print(f"Embedding {i+1}/{len(text_list)}")

#         try:
#             response = genai.embed_content(
#                 model=EMBED_MODEL,
#                 content=text,
#                 task_type="retrieval_document"
#             )

#             # Gemini returns response["embedding"], not ["embeddings"]
#             if "embedding" in response:
#                 embeddings.append(response["embedding"])
#             else:
#                 print("❌ No embedding returned for item:", i)
#                 embeddings.append([0]*768)  # fallback vector

#         except Exception as e:
#             print(f"❌ Error embedding item {i}: {e}")
#             embeddings.append([0]*768)

#     return embeddings

# # ----------------------------------------------------------
# # 2. BUILD FAISS INDEX
# # ----------------------------------------------------------
# def build_faiss_index(chunks, embeddings):
#     if not embeddings:
#         raise ValueError("❌ No embeddings generated — cannot build FAISS index.")

#     dimension = len(embeddings[0])
#     index = faiss.IndexFlatL2(dimension)

#     vectors_np = np.array(embeddings, dtype="float32")
#     index.add(vectors_np)

#     metadata = [
#         {
#             "chunk_id": i,
#             "file_name": chunks[i]["file_name"],
#             "text": chunks[i]["text"]
#         }
#         for i in range(len(chunks))
#     ]

#     return index, metadata


# # ----------------------------------------------------------
# # 3. SAVE VECTOR STORE
# # ----------------------------------------------------------
# def save_vector_store(index, metadata):
#     os.makedirs("vector_store", exist_ok=True)

#     faiss.write_index(index, "vector_store/faiss.index")

#     with open("vector_store/meta.json", "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=2, ensure_ascii=False)

#     print("\n✅ Vector store saved successfully.")


# # ----------------------------------------------------------
# # 4. MAIN PIPELINE
# # ----------------------------------------------------------
# def run_embedding_pipeline():
#     from load_and_chunk import load_and_chunk

#     print("\n🔍 Loading chunks from dataset...")
#     chunks = load_and_chunk("C:/Users/CV MURALI/Documents/TASK_INERG_IR/IR_SYSTEM/Data")
#     print(f"Total chunks to embed: {len(chunks)}")

#     texts = [c["text"] for c in chunks]

#     print("\n🚀 Generating embeddings from Gemini...")
#     embeddings = embed_text_batch(texts)

#     print(f"\n✔ Total embeddings generated: {len(embeddings)}")

#     print("\n📦 Building FAISS vector index...")
#     index, metadata = build_faiss_index(chunks, embeddings)

#     print("\n💾 Saving vector database...")
#     save_vector_store(index, metadata)

#     print("\n🎉 ALL DONE — FAISS index + metadata created successfully.")


# if __name__ == "__main__":
#     run_embedding_pipeline()


import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------- CONFIG -----------------
LOCAL_MODEL_PATH = r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\models\bge-small-en-v1.5"
DATASET_PATH     = r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\IR_SYSTEM\Data"
VECTOR_STORE_DIR = "vector_store"


# ----------------- LOAD MODEL -----------------
print("📌 Loading BGE-small-en-v1.5 embedding model from local folder...")
model = SentenceTransformer(LOCAL_MODEL_PATH)
print("✅ Model loaded successfully!")


# ----------------- 1. EMBEDDING FUNCTION -----------------
def embed_text_batch(text_list, batch_size=128):
    embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        print(f"   → Embedding batch {i} to {i+len(batch)}")

        batch_emb = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings.extend(batch_emb)

    return np.array(embeddings, dtype="float32")


# ----------------- 2. FAISS INDEX CREATION -----------------
def build_faiss_index(chunks, embeddings):

    if embeddings.shape[0] == 0:
        raise ValueError("❌ No embeddings generated!")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    metadata = [
        {
            "chunk_id": i,
            "file_name": chunks[i]["file_name"],
            "text": chunks[i]["text"]
        }
        for i in range(len(chunks))
    ]

    return index, metadata


# ----------------- 3. SAVE FAISS + META -----------------
def save_vector_store(index, metadata):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    faiss.write_index(index, f"{VECTOR_STORE_DIR}/faiss.index")

    with open(f"{VECTOR_STORE_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ Vector store saved successfully!")


# ----------------- 4. MAIN PIPELINE -----------------
def run_embedding_pipeline():
    from load_and_chunk import load_and_chunk

    print("\n🔍 Loading & chunking documents...")
    chunks = load_and_chunk(DATASET_PATH)
    print(f"📌 Total Chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]

    print("\n🚀 Generating embeddings...")
    emb = embed_text_batch(texts)
    print(f"✔ Total embeddings generated: {emb.shape[0]}")

    print("\n📦 Building FAISS index...")
    index, metadata = build_faiss_index(chunks, emb)

    print("\n💾 Saving FAISS + metadata...")
    save_vector_store(index, metadata)

    print("\n🎉 DONE! Vector DB is ready.")


# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    run_embedding_pipeline()
