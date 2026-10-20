"""
เฟส 1 / ด่าน 2+3+4 - ดูข้างในโมเดล: Embedding, Attention, Logits
รัน:  python phase1/02_inside_model.py

สิ่งที่จะได้เห็น:
- token id -> เวกเตอร์ (embedding)
- attention shape: แต่ละคำให้น้ำหนักกับคำอื่นเท่าไหร่
- logits: โมเดลทำนายคำถัดไปว่าน่าจะเป็นอะไร
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,            # ครึ่ง precision ประหยัด VRAM
    device_map="cuda",
    attn_implementation="eager",   # สำคัญ! ต้องใช้ eager ถึงจะดู attention ได้
)                                  # (ค่า default คือ sdpa ซึ่งเร็วกว่าแต่ไม่คืน attention)

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# ----- [2] Embedding: token id -> vector -----
embedding_layer = model.get_input_embeddings()
embeds = embedding_layer(inputs.input_ids)
print(f"Input shape (1, จำนวน_token): {inputs.input_ids.shape}")
print(f"Embedding shape: {embeds.shape}")   # (1, n_tokens, hidden_size)
print(f"-> แต่ละ token กลายเป็นเวกเตอร์ขนาด {embeds.shape[-1]} มิติ")

# ----- [3+4] รันทั้งโมเดล (forward pass ครั้งเดียว) + ขอ attention ออกมาดู -----
# หมายเหตุ: ต้องใช้ model(...) ไม่ใช่ model.generate(...)
#   - model(...)          = forward รอบเดียว -> ได้ .logits, .attentions (ส่องข้างใน)
#   - model.generate(...) = วนสร้างข้อความ   -> ได้แค่ Tensor ของ token ids
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

print(f"\nจำนวน layer: {len(out.attentions)}")
print(f"Attention shape ต่อ layer: {out.attentions[0].shape}")
# (batch, heads, n_tokens, n_tokens) = แต่ละคำให้น้ำหนักกับคำอื่นเท่าไหร่

# logits = คะแนนดิบของทุกคำใน vocab สำหรับ token ถัดไป
print(f"\nLogits shape: {out.logits.shape}")   # (1, n_tokens, vocab_size)
next_token_logits = out.logits[0, -1, :]        # เอาเฉพาะตำแหน่งสุดท้าย
top5 = torch.topk(next_token_logits, 5)
print(f"\n5 คำที่โมเดลคิดว่าน่าจะตามหลัง '{text}':")
for score, idx in zip(top5.values, top5.indices):
    print(f"   {tokenizer.decode([idx]):20s}  (score: {score:.2f})")

# (โบนัส) ดู attention เฉลี่ยของ layer สุดท้าย
print("\n(โบนัส) attention เฉลี่ยทุก head ของ layer สุดท้าย:")
last_attn = out.attentions[-1][0].mean(dim=0)   # (n_tokens, n_tokens)
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
print("tokens:", tokens)
print(last_attn)
