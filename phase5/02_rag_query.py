"""
เฟส 5 / ขั้น 2 - RAG: ถามคำถาม -> ค้นหา -> ตอบพร้อมอ้างอิง
รัน:  python phase5/02_rag_query.py

ขั้นตอน RAG ตอน query:
1. Embed คำถาม -> เวกเตอร์
2. Retrieve   - ค้น chunk ที่ใกล้เคียงที่สุดจาก FAISS (top-k)
3. Augment    - เอา chunk ที่ค้นได้ใส่เข้าไปใน prompt
4. Generate   - ให้ LLM ตอบโดยอิงเฉพาะข้อมูลที่ให้ (ลดการมั่ว/hallucination)

** จุดแข็งของ RAG: อัปเดตความรู้แค่แก้เอกสาร ไม่ต้องเทรนใหม่ **
** เหมาะกับ 8GB มาก เพราะ embedding model เล็ก + ใช้ base model ตอบได้เลย **
"""
import json
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_MODEL = "intfloat/multilingual-e5-small"
LLM = "Qwen/Qwen2.5-0.5B-Instruct"   # หรือใช้ phase3/merged ถ้า fine-tune แล้ว
TOP_K = 2

# โหลด index + chunks + embedder + LLM
index = faiss.read_index("phase5/data/index.faiss")
with open("phase5/data/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LLM)
model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float16, device_map="cuda")


def retrieve(question, k=TOP_K):
    """ค้น chunk ที่เกี่ยวข้องที่สุด"""
    q_emb = embedder.encode([f"query: {question}"], normalize_embeddings=True)
    scores, idx = index.search(q_emb, k)
    return [(chunks[i], float(s)) for i, s in zip(idx[0], scores[0])]


def rag_answer(question):
    # 1-2. Retrieve
    hits = retrieve(question)
    context = "\n\n".join(f"[แหล่งที่ {i+1}]\n{c}" for i, (c, _) in enumerate(hits))

    # 3. Augment - สร้าง prompt ที่บังคับให้ตอบจาก context เท่านั้น
    system = ("ตอบคำถามโดยอิงจากข้อมูลที่ให้เท่านั้น "
              "ถ้าข้อมูลไม่พอให้บอกว่าไม่ทราบ อย่าเดา")
    user = f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    # 4. Generate
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer, hits


questions = [
    "บริษัทก่อตั้งปีอะไร",
    "ลาพักร้อนได้กี่วัน",
    "What are the working hours?",
    "เมืองหลวงของฝรั่งเศสคือ",   # <-- ไม่มีในเอกสาร ควรตอบว่าไม่ทราบ
]
for q in questions:
    answer, hits = rag_answer(q)
    print(f"\n{'='*60}\nคำถาม: {q}")
    print(f"ค้นเจอ (คะแนนความใกล้เคียง): {[f'{s:.2f}' for _, s in hits]}")
    print(f"คำตอบ: {answer}")
