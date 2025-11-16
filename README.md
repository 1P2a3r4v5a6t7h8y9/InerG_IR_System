# InerG_IR_System-Partially Completed


This README provides a detailed explanation of the **Data Acquisition → Text Preprocessing → Sliding Window Chunking → Embedding → Indexing → Retrieval** workflow, including **mathematical concepts, formulas, and tools used**.

---

## 🏛️ 1. Data Acquisition (PDF Extraction)
Using **Andhra Pradesh Government PDF documents** as data source.

### **Tools Used**
- **PyMuPDF (`fitz`)** → Extracts raw text from PDF pages.

performs:
1. Open PDF
2. Read page-by-page
3. Extract text using PyMuPDF's text engine
4. Remove noise (e.g., page numbers)

---

## ✨ 2. Text Preprocessing
The `clean_text()` function removes noise and prepares clean text for chunking.

### **Cleaning Steps**
- Remove non-ASCII characters
- Remove page numbers
- Fix hyphenated line breaks
- Flatten multi-line paragraphs
- Remove extra spaces

This step ensures embeddings capture **semantic meaning** instead of PDF formatting artifacts.

---

## 🧩 3. Sliding Window Text Chunking
Sliding Window ensures **context retention** across chunks.

### **Formula**
Given:
- Sentence list: `S = [S₁, S₂, ..., Sₙ]`
- Window size: `W`
- Step size: `K`

### **Chunk generation rule:**
```
Chunk_i = S[i : i + W]
Next i = i + K
```

### **Number of Chunks**
\[
N = \left\lfloor \frac{n - W}{K} \right\rfloor + 1
\]

### **Why Use Sliding Window?**
✔ Preserves continuity between chunks  
✔ Reduces information loss  
✔ Prevents abrupt context breaks

---

## ⚙️ 4. Embedding (Vectorization)
Using:
**Model:** 'bge-small-en-v1.5' → A sentence embedding model

### **Operation**
Text chunk → Neural network → Dense vector

### **Mathematical Concept**
Each chunk `T` is converted into a **d-dimensional embedding vector**:
\[
E = f(T), \quad E \in \mathbb{R}^d
\]
Where:
- `f` = embedding model
- `d` = vector dimension (e.g., 384)

---

## 🔍 5. Similarity: Cosine Similarity
Cosine similarity is used to match queries with chunks.

### **Formula:**
\[
\text{cos}(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
\]
Where:
- `A`, `B` = embedding vectors

### **Range:**
```
1.0 → identical
0.0 → no relation
-1.0 → opposite meaning
```

---

## ⚡ 6. Vector Indexing (FAISS)
FAISS is used to store and retrieve vectors efficiently.

### **Operation:**
1. Build FAISS index
2. Add embedding vectors
3. Search top‑k nearest neighbors

### **FAISS Search**
Given a query vector `q` and database vectors `xᵢ`:
\[
\text{Find } k \text{ vectors } xᵢ \text{ minimizing } \| q - xᵢ \|
\]
(FAISS internally approximates nearest neighbors for speed.)

---

## 🔄 7. Retrieval Pipeline
1. User query → embedding `E_q`
2. FAISS finds nearest vectors
3. Retrieve corresponding text chunks
4. Pass to LLM for final answer generation

---

## 🧱 Workflow Summary
```
PDF → Extract Text → Clean Text → Sliding Window Chunking → Embedding → Vector Indexing → Retrieval
```

---

## Files Explained
- PDF extraction
- Cleaning
- Smart chunking with sliding window
- Processing all PDFs in a dataset folder


# 📐 Mathematical Concepts Used

| Concept | Formula | Purpose |
|--------|----------|---------|
| Cosine Similarity | \( \frac{q\cdot d}{\|q\|\|d\|} \) | Measure semantic similarity |
| Vector Normalization | \( \frac{v}{\|v\|} \) | Needed for cosine = dot product |
| Sliding Window | \( T[i:i+W] \) | Preserve context between chunks |
| ANN Search | \( \max(q \cdot d_i) \) | Efficient large-scale retrieval |
| Inner Product | \( q \cdot d \) | Fast similarity for FAISS |

----------------------------------------------------------------------------------------------

# ✨ Tools Used & Their Roles

| Tool | Purpose |
|------|---------|
| **PyMuPDF** | PDF text extraction |
| **Regex** | Cleaning, removing noise |
| **BGE Embedding Model** | Converts text → vectors |
| **FAISS** | Vector storage + fast ANN search |
| **Streamlit** | Frontend UI |
| **NumPy** | Vector math & normalization |



# 🟦 Conclusion

In this README document I tried to  complete Information System including :  
✔ Data extraction  
✔ Preprocessing  
✔ Smart chunking  
✔ Sliding window  
✔ Embedding mathematics  
✔ FAISS indexing  
✔ Retrieval & re-ranking  


#Time Constraint Challenges
----------------------------------------------------------------------
Completing everything (data → embedding → retrieval) in a short time.

Testing multiple chunking strategies quickly.

Debugging pipeline failures under time pressure.
