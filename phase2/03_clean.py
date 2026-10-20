"""
เฟส 2 / ขั้น 3 - ทำความสะอาดข้อมูล
รัน:  python phase2/03_clean.py
อ่าน:  phase2/data/raw.jsonl  ->  เขียน:  phase2/data/clean.jsonl

กฎทำความสะอาดที่ใช้บ่อย:
1. strip ช่องว่างหัว-ท้าย + ยุบช่องว่างซ้ำ
2. ลบแถวที่ user หรือ assistant ว่างเปล่า
3. ลบแถวที่สั้นเกินไป (น่าจะไม่มีสาระ)
4. ลบแถวซ้ำ (ดูจาก user+assistant)
"""
import json
import re

MIN_USER_CHARS = 3       # คำถามต้องยาวอย่างน้อยกี่ตัวอักษร
MIN_ASSISTANT_CHARS = 5  # คำตอบต้องยาวอย่างน้อยกี่ตัวอักษร


def normalize(text):
    """strip + ยุบช่องว่าง/บรรทัดซ้ำให้เหลืออันเดียว"""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)      # ยุบ space/tab ซ้ำ
    text = re.sub(r"\n{3,}", "\n\n", text)   # ยุบบรรทัดว่างเกิน
    return text


def get_turn(messages, role):
    """ดึง content ของ role ที่ต้องการ (user/assistant) ออกมา"""
    for m in messages:
        if m["role"] == role:
            return m["content"]
    return ""


rows_in, kept, seen = 0, [], set()
removed = {"empty": 0, "too_short": 0, "duplicate": 0}

with open("phase2/data/raw.jsonl", encoding="utf-8") as f:
    for line in f:
        rows_in += 1
        row = json.loads(line)

        user = normalize(get_turn(row["messages"], "user"))
        assistant = normalize(get_turn(row["messages"], "assistant"))

        # กฎ 2: ว่างเปล่า
        if not user or not assistant:
            removed["empty"] += 1
            continue
        # กฎ 3: สั้นเกินไป
        if len(user) < MIN_USER_CHARS or len(assistant) < MIN_ASSISTANT_CHARS:
            removed["too_short"] += 1
            continue
        # กฎ 4: ซ้ำ
        key = (user, assistant)
        if key in seen:
            removed["duplicate"] += 1
            continue
        seen.add(key)

        kept.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})

with open("phase2/data/clean.jsonl", "w", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"อ่านเข้า:  {rows_in} แถว")
print(f"ลบออก:    ว่าง={removed['empty']}, สั้นเกิน={removed['too_short']}, ซ้ำ={removed['duplicate']}")
print(f"เหลือ:     {len(kept)} แถว -> phase2/data/clean.jsonl")
