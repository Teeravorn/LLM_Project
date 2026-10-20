"""
เฟส 4 / ขั้น 1 - วัด Loss และ Perplexity บนชุด validation
รัน:  python phase4/01_eval_perplexity.py

แนวคิด:
- Loss = โมเดล "งง" กับคำตอบที่ถูกต้องแค่ไหน (ต่ำ = ดี)
- Perplexity = exp(loss) = "โดยเฉลี่ยโมเดลลังเลระหว่างกี่ตัวเลือก"
  perplexity 1 = มั่นใจเป๊ะ, ยิ่งสูงยิ่งงง
- วัดบน val set (ข้อมูลที่โมเดลไม่เคยเห็นตอนเทรน) = วัดการเรียนรู้จริง
"""
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "phase3/adapter"
VAL_FILE = "phase2/data/val.jsonl"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()


def perplexity_of(messages):
    """คำนวณ perplexity ของตัวอย่างเดียว"""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=enc.input_ids)  # labels=input_ids -> LM loss
    return out.loss.item(), enc.input_ids.size(1)


total_loss, total_tokens = 0.0, 0
with open(VAL_FILE, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        loss, n = perplexity_of(row["messages"])
        total_loss += loss * n
        total_tokens += n
        print(f"  ตัวอย่าง: loss={loss:.3f}  perplexity={math.exp(loss):.2f}")

avg_loss = total_loss / total_tokens
print(f"\n{'='*50}")
print(f"Loss เฉลี่ย (val):      {avg_loss:.4f}")
print(f"Perplexity เฉลี่ย (val): {math.exp(avg_loss):.2f}")
print("-> เก็บตัวเลขนี้ไว้เทียบก่อน/หลังเทรน หรือระหว่าง epoch")
print("-> ถ้า train loss ลงเรื่อย ๆ แต่ val loss เริ่มขึ้น = overfitting!")
