"""
เฟส 1 / ด่าน 5 - Sampling: ทำไมตอบไม่เหมือนเดิมทุกครั้ง
รัน:  python phase1/03_sampling.py

แนวคิด: logits -> softmax เป็นความน่าจะเป็น -> เลือกคำถัดไป
วิธีเลือก (temperature, top_p, top_k) มีผลกับ "บุคลิก" คำตอบ
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda"
)

# ใช้ chat template (สำคัญ! โมเดล instruct ต้องการ format นี้)
messages = [{"role": "user", "content": "อธิบายคำว่า AI สั้น ๆ"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("=== temperature=0 (greedy: เลือกคำมั่นใจสุดเสมอ -> ตอบเหมือนเดิม) ===")
out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

print("\n=== temperature=0.9 (สุ่มมากขึ้น -> สร้างสรรค์/หลากหลาย) ===")
out = model.generate(
    **inputs, max_new_tokens=80,
    do_sample=True, temperature=0.9, top_p=0.9,  # top_p = nucleus sampling
)
print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

print("\n=== ลองเล่น: เปลี่ยน temperature เป็น 0.1, 0.7, 1.5 ดูความต่าง ===")
for temp in [0.1, 0.7, 1.5]:
    out = model.generate(
        **inputs, max_new_tokens=60,
        do_sample=True, temperature=temp, top_p=0.9,
    )
    answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n--- temperature={temp} ---")
    print(answer)
