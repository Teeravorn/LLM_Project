"""
เฟส 3 / ขั้น 1 - Fine-tune ด้วย QLoRA (4-bit + LoRA)
รัน:  python phase3/01_train_qlora.py
ใช้ข้อมูล:  phase2/data/train.jsonl, phase2/data/val.jsonl  (จากเฟส 2)
ผลลัพธ์:    phase3/adapter/  (เก็บแค่ adapter ตัวเล็ก ไม่ใช่ทั้งโมเดล)

แนวคิด QLoRA:
  - โหลดโมเดลแบบ 4-bit (ประหยัด VRAM 4 เท่า)
  - แช่แข็งน้ำหนักเดิมทั้งหมด เทรนแค่ LoRA adapter (~1% ของพารามิเตอร์)
  - เทรน 7B บน 8GB ได้ (0.5B ที่ใช้ตรงนี้สบายมาก)

** หมายเหตุ: ข้อมูลตัวอย่างจากเฟส 2 มีแค่ไม่กี่แถว ใช้เพื่อ "ดูว่า pipeline
   ทำงานครบ" เท่านั้น ผลลัพธ์จะยังไม่เก่ง งานจริงต้องมีข้อมูลหลายร้อย-พันแถว **
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # เริ่มตัวเล็ก พอคล่องแล้วเปลี่ยนเป็น 3B/7B ได้

# ===== 1. โหลดโมเดลแบบ 4-bit (หัวใจของ QLoRA) =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # nf4 = แม่นกว่า int4 ธรรมดา
    bnb_4bit_compute_dtype=torch.float16,   # คำนวณด้วย fp16
    bnb_4bit_use_double_quant=True,         # บีบซ้ำอีกชั้น ประหยัดเพิ่ม
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False   # ต้องปิดตอนเทรน (ใช้คู่กับ gradient checkpointing)

# ===== 2. ตั้งค่า LoRA adapter =====
peft_config = LoraConfig(
    r=16,                # ขนาด adapter
    lora_alpha=32,       # = 2 * r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # ชั้นที่จะแปะ adapter: attention (q,k,v,o) + mlp (gate,up,down)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ===== 3. โหลดข้อมูลจากเฟส 2 =====
dataset = load_dataset("json", data_files={
    "train": "phase2/data/train.jsonl",
    "validation": "phase2/data/val.jsonl",
})
# dataset มีคอลัมน์ "messages" -> TRL จะ apply_chat_template ให้อัตโนมัติ

# ===== 4. ตั้งค่าการเทรน (ปรับให้พอดี 8GB) =====
args = SFTConfig(
    output_dir="phase3/output",
    num_train_epochs=3,
    per_device_train_batch_size=1,      # 8GB ต้องคุม batch=1
    gradient_accumulation_steps=8,      # สะสม 8 step = เสมือน batch 8
    gradient_checkpointing=True,        # แลกความเร็วกับ VRAM (ประหยัดมาก)
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    eval_strategy="epoch",              # ประเมินด้วย val ทุก epoch
    save_strategy="epoch",
    max_length=512,                     # ตั้งตามสถิติ token จากเฟส 2
    report_to="none",                   # ไม่ส่งไป wandb
)

# ===== 5. เทรน! =====
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
)

# แสดงว่าเทรนแค่กี่ % ของพารามิเตอร์ทั้งหมด (จะเห็นว่าน้อยมาก)
trainer.model.print_trainable_parameters()

trainer.train()

# ===== 6. เซฟ adapter (ตัวเล็ก ไม่กี่ MB) =====
trainer.save_model("phase3/adapter")
tokenizer.save_pretrained("phase3/adapter")
print("\n✅ เทรนเสร็จ! adapter อยู่ที่ phase3/adapter")
print("ขั้นต่อไป: python phase3/02_inference.py")
