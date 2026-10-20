# เฟส 6 / ทางเลือก - รันโมเดลด้วย Ollama (ง่ายที่สุดสำหรับใช้งานจริง)

Ollama = โปรแกรมรันโมเดลบนเครื่องแบบง่าย ๆ จัดการ quantization/VRAM ให้อัตโนมัติ
เหมาะกับ RTX 4060 8GB มาก เพราะมันเลือก quantization ที่พอดี VRAM ให้เอง

## วิธีที่ 1: ใช้โมเดลสำเร็จรูป (ง่ายสุด เริ่มที่นี่)

1. ติดตั้ง Ollama จาก https://ollama.com/download (มีตัวติดตั้ง Windows)
2. เปิด PowerShell แล้วรัน:

```powershell
# โหลด + รันโมเดล (4-bit อัตโนมัติ พอดี 8GB)
ollama run qwen2.5:7b

# หรือตัวเล็กกว่า เร็วกว่า
ollama run qwen2.5:3b
```

3. พิมพ์คุยได้เลย หรือเรียกผ่าน API ที่ http://localhost:11434

## วิธีที่ 2: ใช้โมเดลที่เรา fine-tune เอง (จากเฟส 3)

ต้องแปลง adapter -> merged -> GGUF ก่อน:

```powershell
# 1. merge adapter เข้า base (จากเฟส 3)
python phase3/03_merge.py        # ได้ phase3/merged/

# 2. แปลงเป็น GGUF ด้วย llama.cpp
#    โคลน: git clone https://github.com/ggerganov/llama.cpp
#    แล้วรัน convert:
python llama.cpp/convert_hf_to_gguf.py phase3/merged --outfile my-model.gguf --outtype q4_k_m
```

3. สร้างไฟล์ชื่อ `Modelfile` (ดูตัวอย่างในโฟลเดอร์นี้) แล้ว:

```powershell
ollama create my-llm -f phase6/Modelfile
ollama run my-llm
```

## เรียก Ollama ผ่าน Python

```python
import requests
r = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:3b",
    "prompt": "AI คืออะไร",
    "stream": False,
})
print(r.json()["response"])
```

## เทียบ FastAPI (ขั้น 01) vs Ollama

| | FastAPI (01_fastapi_server.py) | Ollama |
|---|---|---|
| ควบคุมได้ละเอียด | ✅ มาก | ปานกลาง |
| ตั้งค่าง่าย | ต้องเขียนเอง | ✅ ง่ายมาก |
| จัดการ VRAM/quantize | ทำเอง | ✅ อัตโนมัติ |
| เหมาะกับ | เรียนรู้ + ปรับแต่ง | ใช้งานจริงเร็ว ๆ |
