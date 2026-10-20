"""
เฟส 3 / ขั้น 3 (ทางเลือก) - รวม adapter เข้ากับโมเดล base ให้เป็นโมเดลเดียว
รัน:  python phase3/03_merge.py
ผลลัพธ์:  phase3/merged/  (โมเดลเต็มที่รวม adapter แล้ว)

ทำไมต้อง merge:
  - ตอนเทรน: เก็บแค่ adapter (เล็ก, แชร์ง่าย) = ดี
  - ตอน deploy จริง: อยาก export เป็น GGUF / ใช้กับ Ollama (เฟส 6)
    ต้อง merge เป็นโมเดลเดียวก่อน

หมายเหตุ: ตอน merge ต้องโหลดแบบ fp16 (ไม่ใช่ 4-bit) เพื่อให้รวมน้ำหนักได้แม่นยำ
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "phase3/adapter"
OUT = "phase3/merged"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

# โหลด base แบบ fp16 (ไม่ quantize) เพื่อ merge ให้แม่น
base = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cpu"
)
model = PeftModel.from_pretrained(base, ADAPTER)

# รวมน้ำหนัก adapter เข้าไปในโมเดลหลัก
model = model.merge_and_unload()

model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print(f"✅ merge เสร็จ! โมเดลเต็มอยู่ที่ {OUT}")
print("เอาไปใช้ต่อในเฟส 6 (export GGUF / Ollama) ได้เลย")
