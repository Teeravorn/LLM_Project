"""
เฟส 2 / ขั้น 2 - สร้างชุดข้อมูล "ดิบ" (จงใจใส่ noise เพื่อฝึกทำความสะอาด)
รัน:  python phase2/02_build_raw.py
ผลลัพธ์:  phase2/data/raw.jsonl

ในงานจริงข้อมูลดิบมาจาก: ไฟล์ที่เราเขียนเอง, export จากแชท, scrape เว็บ,
dataset สาธารณะ (HuggingFace) ฯลฯ ซึ่งมักจะมีปัญหา:
ซ้ำ, ช่องว่างเกิน, ว่างเปล่า, สั้น/ยาวผิดปกติ
"""
import json
import os

os.makedirs("phase2/data", exist_ok=True)

# ข้อมูลดิบ (มี noise จงใจ) -- ปกติส่วนนี้มาจากการรวบรวมจริง
raw = [
    {"messages": [
        {"role": "user", "content": "AI คืออะไร"},
        {"role": "assistant", "content": "AI คือปัญญาประดิษฐ์ ระบบที่ทำให้คอมพิวเตอร์เรียนรู้และตัดสินใจได้"}]},
    {"messages": [
        {"role": "user", "content": "AI คืออะไร"},   # <-- ซ้ำกับอันบน (ต้องลบ)
        {"role": "assistant", "content": "AI คือปัญญาประดิษฐ์ ระบบที่ทำให้คอมพิวเตอร์เรียนรู้และตัดสินใจได้"}]},
    {"messages": [
        {"role": "user", "content": "   What is machine learning?   "},  # <-- ช่องว่างเกิน
        {"role": "assistant", "content": "Machine learning is a subset of AI where systems learn patterns from data."}]},
    {"messages": [
        {"role": "user", "content": "สวัสดี"},
        {"role": "assistant", "content": ""}]},   # <-- คำตอบว่าง (ต้องลบ)
    {"messages": [
        {"role": "user", "content": "ก"},          # <-- สั้นเกินไป (ต้องลบ)
        {"role": "assistant", "content": "ครับ"}]},
    {"messages": [
        {"role": "user", "content": "อธิบาย neural network แบบเข้าใจง่าย"},
        {"role": "assistant", "content": "Neural network คือโมเดลที่เลียนแบบการทำงานของสมอง "
         "ประกอบด้วยชั้นของ node เชื่อมต่อกัน แต่ละ node รับ input คูณน้ำหนัก แล้วส่งต่อ"}]},
    {"messages": [
        {"role": "user", "content": "How do I say thank you in Thai?"},
        {"role": "assistant", "content": "You say 'ขอบคุณ' (khob khun)."}]},
    {"messages": [
        {"role": "user", "content": "RAG ย่อมาจากอะไร"},
        {"role": "assistant", "content": "RAG ย่อมาจาก Retrieval-Augmented Generation "
         "คือเทคนิคให้โมเดลค้นข้อมูลก่อนตอบ"}]},
]

with open("phase2/data/raw.jsonl", "w", encoding="utf-8") as f:
    for row in raw:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"เขียน {len(raw)} แถว -> phase2/data/raw.jsonl")
print("(มี noise จงใจ: ซ้ำ 1, ช่องว่างเกิน 1, คำตอบว่าง 1, สั้นเกิน 1)")
print("ขั้นต่อไป: รัน 03_clean.py เพื่อทำความสะอาด")
