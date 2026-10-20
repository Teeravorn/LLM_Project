"""
เฟส 6 / ขั้น 1 - เสิร์ฟโมเดลเป็น API ด้วย FastAPI (มี streaming)
เตรียม:  pip install fastapi uvicorn
รัน:     python phase6/01_fastapi_server.py
เปิด:    http://127.0.0.1:8000/docs  (ลองยิง API ผ่านหน้าเว็บได้เลย)

แนวคิด: หุ้มโมเดลด้วย HTTP API เพื่อให้แอปอื่น (เว็บ/มือถือ/บอท) เรียกใช้ได้
streaming = ส่งคำตอบทีละ token เหมือน ChatGPT (ไม่ต้องรอจบทั้งประโยค)
"""
import threading
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import uvicorn

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # หรือ phase3/merged ถ้า fine-tune แล้ว

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")

app = FastAPI(title="LLM API")


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 128
    temperature: float = 0.7


@app.post("/chat")
def chat(req: ChatRequest):
    """ตอบแบบ streaming ทีละ token"""
    messages = [{"role": "user", "content": req.message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs, streamer=streamer, max_new_tokens=req.max_tokens,
        do_sample=True, temperature=req.temperature, top_p=0.9,
    )
    # generate ใน thread แยก เพื่อให้ streamer ทยอยส่งออกมาได้
    threading.Thread(target=model.generate, kwargs=gen_kwargs).start()

    def token_stream():
        for token in streamer:
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
