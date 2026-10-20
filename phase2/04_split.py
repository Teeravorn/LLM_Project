"""
เฟส 2 / ขั้น 4 - แบ่ง train/val + ตรวจสอบสถิติ token
รัน:  python phase2/04_split.py
อ่าน:  phase2/data/clean.jsonl
เขียน: phase2/data/train.jsonl, phase2/data/val.jsonl

ทำไมต้องแบ่ง: เก็บ val ไว้ "สอบ" โมเดล เพื่อดูว่าเรียนรู้จริงหรือท่องจำ (overfitting)
ทำไมเช็ค token: ภาษาไทยกิน token เยอะ ต้องรู้ความยาวสูงสุดเพื่อตั้ง max_length ตอนเทรน
"""
import json
import random
from transformers import AutoTokenizer

VAL_RATIO = 0.2   # แบ่ง 20% ไว้ validate
SEED = 42         # fix seed เพื่อให้แบ่งเหมือนเดิมทุกครั้ง (ทำซ้ำได้)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# อ่านข้อมูลสะอาด
rows = []
with open("phase2/data/clean.jsonl", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# สับไพ่ก่อนแบ่ง (สำคัญ! กันข้อมูลเรียงเป็นกลุ่ม)
random.seed(SEED)
random.shuffle(rows)

n_val = max(1, int(len(rows) * VAL_RATIO))
val_rows = rows[:n_val]
train_rows = rows[n_val:]

for name, data in [("train", train_rows), ("val", val_rows)]:
    with open(f"phase2/data/{name}.jsonl", "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"ทั้งหมด {len(rows)} แถว -> train {len(train_rows)}, val {len(val_rows)}")

# ----- ตรวจสอบสถิติความยาว token -----
lengths = []
for row in rows:
    n = len(tokenizer.apply_chat_template(row["messages"]))
    lengths.append(n)

lengths.sort()
print("\nสถิติความยาว (token ต่อตัวอย่าง):")
print(f"   สั้นสุด: {lengths[0]}")
print(f"   ยาวสุด:  {lengths[-1]}")
print(f"   เฉลี่ย:  {sum(lengths)/len(lengths):.1f}")
print(f"   -> ตอนเทรนควรตั้ง max_length >= {lengths[-1]} (เผื่อให้ครอบคลุมตัวที่ยาวสุด)")
