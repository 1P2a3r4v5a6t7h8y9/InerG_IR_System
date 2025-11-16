import os
import re
import fitz
import warnings

# Disable noisy warnings from MuPDF
warnings.filterwarnings("ignore", category=UserWarning)
fitz.TOOLS.mupdf_display_errors(False)


# ------------ 1. PDF LOADER (SAFE VERSION) ----------------
def extract_pdf(file_path):
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"❌ Failed to open {file_path}: {e}")
        return ""  # Return empty text so script continues

    text = ""

    for page_num, page in enumerate(doc, start=1):
        try:
            page_text = page.get_text()
        except Exception as e:
            print(f"⚠️ Failed reading page {page_num} in {file_path}: {e}")
            continue

        # Remove standalone page numbers (common in gov PDFs)
        cleaned = "\n".join([
            line for line in page_text.split("\n")
            if not re.match(r"^\s*\d+\s*$", line)
        ])

        text += cleaned + "\n"

    return text.strip()


# ------------ 2. SMART CHUNKER ----------------
def smart_chunking(text):
    if not text:
        return []

    # Split major sections (e.g., 1. Heading, 2. Heading)
    sections = re.split(r"\n(?=\d+\.\s)", text)

    final_chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", sec)

        temp = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue

            temp.append(s)

            if len(temp) >= 5:  # 5 sentences per chunk
                final_chunks.append(" ".join(temp))
                temp = []

        if temp:
            final_chunks.append(" ".join(temp))

    return final_chunks


# ------------ 3. LOAD ALL PDFs + CHUNK SAFELY ---------------  
def load_and_chunk(folder_path):
    all_chunks = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing: {pdf_path}")

                try:
                    text = extract_pdf(pdf_path)
                    chunks = smart_chunking(text)

                    for ch in chunks:
                        all_chunks.append({
                            "text": ch,
                            "file_name": file
                        })

                    print(f" → {len(chunks)} chunks created.\n")

                except Exception as e:
                    print(f"❌ Error processing {pdf_path}: {e}")
                    continue

    print(f"TOTAL CHUNKS CREATED: {len(all_chunks)}")
    return all_chunks


# ------------ 4. RUN ----------------
if __name__ == "__main__":
    folder = r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\IR_SYSTEM\Data"
    chunks = load_and_chunk(folder)


# import os
# import re
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import warnings

# # Disable noisy MuPDF warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# fitz.TOOLS.mupdf_display_errors(False)

# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\CV MURALI\Documents\tesseract-ocr-w64-setup-5.5.0.20241111.exe"


# # ------------ 1. PDF LOADER WITH OCR ----------------
# def extract_pdf(file_path):
#     try:
#         doc = fitz.open(file_path)
#     except Exception as e:
#         print(f"❌ Failed to open {file_path}: {e}")
#         return ""

#     text = ""

#     for page_num, page in enumerate(doc, start=1):
#         # Extract normal text first
#         try:
#             page_text = page.get_text()
#         except:
#             page_text = ""

#         # Apply OCR if no text present (scanned page)
#         if not page_text.strip():
#             try:
#                 pix = page.get_pixmap(dpi=300)
#                 img = Image.open(io.BytesIO(pix.tobytes()))
#                 page_text = pytesseract.image_to_string(img, lang="eng")
#                 print(f"🔍 OCR applied on page {page_num}")
#             except Exception as e:
#                 print(f"⚠️ OCR failed on page {page_num}: {e}")
#                 continue

#         # Clean out standalone page numbers (common in govt PDFs)
#         cleaned = "\n".join([
#             line for line in page_text.split("\n")
#             if not re.match(r"^\s*\d+\s*$", line.strip())
#         ])

#         text += cleaned + "\n"

#     return text.strip()


# # ------------ 2. SECTION + SENTENCE CHUNKER ----------------
# def smart_chunking(text):
#     if not text:
#         return []

#     # Split major numbered sections (1. Title / 2. Title...)
#     sections = re.split(r"\n(?=\d+\.\s)", text)

#     final_chunks = []

#     for sec in sections:
#         sec = sec.strip()
#         if not sec:
#             continue

#         # Split into sentences
#         sentences = re.split(r"(?<=[.!?])\s+", sec)

#         temp = []
#         for s in sentences:
#             s = s.strip()
#             if not s:
#                 continue

#             temp.append(s)

#             # 5 sentences per chunk
#             if len(temp) >= 5:
#                 final_chunks.append(" ".join(temp))
#                 temp = []

#         if temp:
#             final_chunks.append(" ".join(temp))

#     return final_chunks


# # ------------ 3. LOAD PDFs + RETURNS CHUNK DICTIONARIES ---------------
# def load_and_chunk(folder_path):
#     all_chunks = []

#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(".pdf"):
#                 pdf_path = os.path.join(root, file)
#                 print(f"\n📄 Processing: {pdf_path}")

#                 text = extract_pdf(pdf_path)
#                 chunks = smart_chunking(text)

#                 print(f" → {len(chunks)} chunks created.")

#                 # Store as structured chunks for FAISS/Gemini embedding
#                 for ch in chunks:
#                     all_chunks.append({
#                         "text": ch,
#                         "file_name": file
#                     })

#     print(f"\n✅ TOTAL CHUNKS CREATED: {len(all_chunks)}")
#     return all_chunks


# # ------------ 4. MAIN RUN (Optional) ---------------
# if __name__ == "__main__":
#     folder = r"C:\Users\CV MURALI\Documents\TASK_INERG_IR\IR_SYSTEM\Data"
#     chunks = load_and_chunk(folder)
