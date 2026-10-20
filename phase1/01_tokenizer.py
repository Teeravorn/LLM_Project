"""
เฟส 1 / ด่าน 1 - Tokenizer: แปลงข้อความเป็นตัวเลข (token ids)
รัน:  python phase1/01_tokenizer.py

สิ่งที่ต้องสังเกต:
- ภาษาไทยใช้ token เยอะกว่าอังกฤษมาก (กระทบ context window + ความเร็ว)
- เครื่องหมาย ▁ / Ġ ข้างหน้า token คือช่องว่าง
"""
from transformers import AutoTokenizer

# ใช้ Qwen2.5 — รองรับไทย+อังกฤษดี และเป็น base ที่เราจะใช้เทรนในเฟสหลัง
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # ตัวเล็ก โหลดไว เหมาะเรียนรู้
tokenizer = AutoTokenizer.from_pretrained(MODEL)

texts = [
    "Hello, world!",
    "สวัสดีครับ ผมกำลังเรียน LLM",
    "What is artificial intelligence",
    "ปัญญาประดิษฐ์คืออะไร",
]

for text in texts:
    ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"\nข้อความ: {text}")
    print(f"จำนวน token: {len(ids)}")
    print(f"Token ids: {ids}")
    print(f"Tokens: {tokens}")
    print(f"Decode กลับ: {tokenizer.decode(ids)}")

print("\n" + "=" * 50)
print("ลองเทียบ: ไทย vs อังกฤษ ความหมายเดียวกัน อันไหนกิน token มากกว่า?")
