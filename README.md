# LLM Learning Project 🚀

โปรเจกต์เรียนรู้ LLM ตั้งแต่ต้นจนจบ ครบทุก stage — สำหรับ RTX 4060 (8GB VRAM)

## Hardware ที่ใช้
- CPU: AMD Ryzen 5 5600X | RAM: 16GB | GPU: RTX 4060 8GB

## หลักคิดสำหรับ 8GB VRAM
- ใช้ **QLoRA** (4-bit + LoRA) เทรนโมเดล ≤7B ได้
- `batch_size=1` + gradient accumulation + gradient checkpointing
- RAG และ embedding รันสบาย

---

## โครงสร้างโปรเจกต์ + ลำดับการเรียน

| เฟส | โฟลเดอร์ | เรียนอะไร |
|-----|----------|-----------|
| 0 | `check_gpu.py`, `requirements.txt` | ตั้ง environment, เช็ค CUDA |
| 1 | `phase1/` | LLM ทำงานยังไง: token, embedding, attention, sampling, temperature |
| 2 | `phase2/` | เตรียมข้อมูล: format, clean, split |
| 3 | `phase3/` | Fine-tune ด้วย QLoRA + inference + merge |
| 4 | `phase4/` | Evaluation: perplexity, เทียบ base/finetuned, LLM-as-judge |
| 5 | `phase5/` | RAG: ingest + query พร้อมอ้างอิง |
| 6 | `phase6/` | Deploy: FastAPI streaming + Ollama |
| 7 | `phase7/` | (โบนัส) เทรน mini-GPT จากศูนย์ เข้าใจ Transformer |

---

## เริ่มต้น (ทำครั้งเดียว)

```powershell
cd D:\LLM_Project
python -m venv venv
.\venv\Scripts\Activate.ps1

# PyTorch + CUDA 12.1 (สำคัญ! ต้องลงก่อน)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
python check_gpu.py          # ต้องได้ CUDA available: True
```

## ลำดับการรัน (ทีละเฟส)

```powershell
# เฟส 1 - เข้าใจ LLM
python phase1\01_tokenizer.py
python phase1\02_inside_model.py
python phase1\03_sampling.py
python phase1\04_temperature_demo.py

# เฟส 2 - เตรียมข้อมูล (รันตามลำดับ)
python phase2\01_explore_format.py
python phase2\02_build_raw.py
python phase2\03_clean.py
python phase2\04_split.py

# เฟส 3 - Fine-tune
python phase3\01_train_qlora.py
python phase3\02_inference.py
python phase3\03_merge.py          # (ทางเลือก) สำหรับ deploy

# เฟส 4 - Evaluate
python phase4\01_eval_perplexity.py
python phase4\02_compare_base_vs_finetuned.py
python phase4\03_llm_as_judge.py   # ต้องมี ANTHROPIC_API_KEY

# เฟส 5 - RAG
python phase5\01_ingest.py
python phase5\02_rag_query.py

# เฟส 6 - Deploy
python phase6\01_fastapi_server.py   # เปิด server (หน้าต่างหนึ่ง)
python phase6\02_client.py           # เรียก API (อีกหน้าต่าง)
#   ดู phase6/OLLAMA_GUIDE.md สำหรับวิธี Ollama

# เฟส 7 - เทรนจากศูนย์ (โบนัส)
python phase7\01_char_gpt.py
```

---

## เมื่อคล่องแล้ว: ก้าวต่อไป
- เปลี่ยน base model เป็น `Qwen/Qwen2.5-7B-Instruct` (เฟส 3)
- ใส่ข้อมูลจริงของคุณเองในเฟส 2 (แทน `02_build_raw.py`)
- โหลด dataset ไทยจาก HuggingFace เช่น `Thaweewat/alpaca-cleaned-52k-th`
- ใส่ PDF จริงเข้า RAG (ใช้ `pypdf` อ่าน แล้ว chunk)
- ลองเทคนิคขั้นสูง: DPO, RAG + reranker, function calling

## Dependencies เพิ่มเติมตามเฟส
- เฟส 4 (judge): `pip install anthropic`
- เฟส 6 (API): `pip install fastapi uvicorn requests`
