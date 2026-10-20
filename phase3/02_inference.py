"""
เฟส 3 / ขั้น 2 - ทดสอบโมเดลที่เทรนแล้ว (base + adapter)
รัน:  python phase3/02_inference.py

แนวคิด: โหลดโมเดล base เดิม แล้ว "แปะ" adapter ที่เราเทรนเข้าไป
adapter ตัวเล็กแค่ไม่กี่ MB แต่เปลี่ยนพฤติกรรมโมเดลได้
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "phase3/adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER)   # แปะ adapter
model.eval()


def chat(question):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128,
            do_sample=True, temperature=0.7, top_p=0.9,
        )
    answer = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return answer


# ลองถามคำถามที่ใกล้เคียงกับข้อมูลที่เทรน
questions = [
    "AI คืออะไร",
    "RAG ย่อมาจากอะไร",
    "How do I say thank you in Thai?",
]
for q in questions:
    print(f"\n{'='*60}\nถาม: {q}")
    print(f"ตอบ: {chat(q)}")
