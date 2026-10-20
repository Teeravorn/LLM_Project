"""
เฟส 1 / โบนัส - Temperature demo: เห็นตัวเลขจริงว่าทำไมเรียก "อุณหภูมิ"
รัน:  python phase1/04_temperature_demo.py

แนวคิด: softmax ที่มี temperature
    P(คำ_i) = exp(logit_i / T) / Σ exp(logit_j / T)

T ต่ำ  -> คะแนนถ่างออก -> ความน่าจะเป็น "แหลม" (มั่นใจคำเดียว)
T สูง  -> คะแนนเกลี่ยกัน -> ความน่าจะเป็น "แบน"  (สุ่มหลากหลาย)

** ไฟล์นี้ใช้แค่ตัวเลขสมมติ ไม่ต้องโหลดโมเดล รันได้ทันที **
"""
import torch

# สมมติว่าหลังประโยค "วันนี้อากาศ..." โมเดลให้คะแนน (logits) 5 คำนี้
words = ["ร้อน", "หนาว", "ดี", "แย่", "ฝนตก"]
logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.2])


def softmax_with_temperature(logits, T):
    """หาร logits ด้วย T ก่อน แล้วค่อย softmax"""
    return torch.softmax(logits / T, dim=-1)


print("Logits (คะแนนดิบ):")
for w, l in zip(words, logits):
    print(f"   {w:8s} = {l:.1f}")

print("\n" + "=" * 60)
print("ความน่าจะเป็นเมื่อใช้ temperature ต่าง ๆ:")
print("=" * 60)

for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"\n--- T = {T} ---")
    for w, p in zip(words, probs):
        bar = "█" * int(p * 40)        # วาดกราฟแท่งให้เห็นภาพ
        print(f"   {w:8s} {p*100:5.1f}%  {bar}")

print("\n" + "=" * 60)
print("สังเกต:")
print("  T ต่ำ (0.1)  -> 'ร้อน' เกือบ 100%  = แหลม/มั่นใจ  (เย็น = เป็นระเบียบ)")
print("  T สูง (5.0)  -> ทุกคำใกล้ ๆ กัน    = แบน/สุ่มมั่ว  (ร้อน = วุ่นวาย)")
print("  T = 1.0      -> ใช้คะแนนตามจริง ไม่บิด")
