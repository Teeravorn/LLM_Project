"""
เฟส 5 / ขั้น 1 - RAG: สร้าง index จากเอกสาร (ingest)
รัน:  python phase5/01_ingest.py
อ่าน:  phase5/data/knowledge.md
เขียน: phase5/data/index.faiss + phase5/data/chunks.json

ขั้นตอน RAG ตอน ingest:
1. Chunking  - หั่นเอกสารเป็นชิ้นเล็ก (ที่นี่หั่นตามหัวข้อ ##)
2. Embedding - แปลงแต่ละ chunk เป็นเวกเตอร์ (ตัวเลขที่จับความหมาย)
3. Index     - เก็บเวกเตอร์ใน FAISS เพื่อค้นหาเร็ว
"""
import json
import faiss
from sentence_transformers import SentenceTransformer

# โมเดล embedding แบบ multilingual (รองรับไทย+อังกฤษ) ตัวเล็ก โหลดไว
EMBED_MODEL = "intfloat/multilingual-e5-small"

# ----- 1. Chunking: หั่นเอกสารตามหัวข้อ ## -----
with open("phase5/data/knowledge.md", encoding="utf-8") as f:
    content = f.read()

chunks = []
for block in content.split("\n## "):
    block = block.strip().lstrip("#").strip()
    if block:
        chunks.append(block)
print(f"หั่นได้ {len(chunks)} chunks")

# ----- 2. Embedding -----
embedder = SentenceTransformer(EMBED_MODEL)
# e5 ต้องเติม prefix "passage: " ให้เอกสาร (และ "query: " ให้คำถาม)
embeddings = embedder.encode(
    [f"passage: {c}" for c in chunks],
    normalize_embeddings=True,   # normalize เพื่อใช้ cosine similarity
)

# ----- 3. เก็บใน FAISS index -----
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # IP + normalized = cosine similarity
index.add(embeddings)

faiss.write_index(index, "phase5/data/index.faiss")
with open("phase5/data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"เก็บ index ({dim} มิติ) และ chunks เรียบร้อย")
print("ขั้นต่อไป: python phase5/02_rag_query.py")
