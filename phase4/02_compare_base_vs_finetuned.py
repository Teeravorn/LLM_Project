"""
เฟส 4 / ขั้น 2 - เทียบโมเดล base (ยังไม่เทรน) vs โมเดลที่ fine-tune แล้ว
รัน:  python phase4/02_compare_base_vs_finetuned.py

วิธีประเมินที่ตรงที่สุด: เอาคำถามชุดเดียวกันถามทั้งสองโมเดล แล้วอ่านเทียบ
(eval เชิงคุณภาพ - บางอย่างตัวเลข loss บอกไม่ได้ ต้องใช้ตาคนดู)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "phase3/adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
finetuned = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto"),
    ADAPTER,
)


def ask(model, question):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)  # greedy = เทียบยุติธรรม
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


questions = ["AI คืออะไร", "RAG ย่อมาจากอะไร"]
for q in questions:
    print(f"\n{'='*60}\nคำถาม: {q}")
    print(f"\n[BASE - ยังไม่เทรน]\n{ask(base, q)}")
    print(f"\n[FINE-TUNED]\n{ask(finetuned, q)}")
