"""
เฟส 7 (โบนัส) - เทรน mini-GPT จากศูนย์ (แบบ nanoGPT, char-level)
รัน:  python phase7/01_char_gpt.py

จุดประสงค์: เข้าใจ architecture ของ Transformer จริง ๆ โดยเขียนเองทั้งหมด
ไม่ใช้ transformers library - เห็นทุกชิ้นส่วน: embedding, attention, block, การเทรน

** โมเดลนี้จิ๋วมาก (~ล้านพารามิเตอร์) เทรนบน RTX 4060 ไม่กี่นาที **
** มันจะเรียน "รูปแบบตัวอักษร" จาก corpus - ผลลัพธ์จะพอมั่ว ๆ แต่เห็นกลไก **

แทน corpus: เอาข้อความยาว ๆ ใส่ใน phase7/data/corpus.txt (ยิ่งเยอะยิ่งดี)
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Hyperparameters ----------
batch_size = 32
block_size = 64          # ดูย้อนหลังกี่ตัวอักษร
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
learning_rate = 3e-3
max_iters = 2000
eval_interval = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ---------- ข้อมูล ----------
CORPUS_PATH = "phase7/data/corpus.txt"
if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        text = f.read()
else:
    # corpus สำรอง (เล็กมาก - แนะนำให้ใส่ข้อความจริงยาว ๆ แทน)
    text = ("ปัญญาประดิษฐ์คือศาสตร์ที่ทำให้เครื่องจักรเรียนรู้ได้ "
            "machine learning is a subset of artificial intelligence. ") * 200

# char-level tokenizer (ง่ายสุด: 1 ตัวอักษร = 1 token)
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------- ชิ้นส่วน Transformer (เขียนเอง) ----------
class Head(nn.Module):
    """self-attention หนึ่งหัว - หัวใจของ Transformer"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # mask สามเหลี่ยม: กันไม่ให้ token มองอนาคต (causal)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        # คะแนน attention: query คูณ key
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v   # ถ่วงน้ำหนัก value ด้วยคะแนน attention


class MultiHead(nn.Module):
    """หลายหัว attention ทำงานขนานกัน"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """ชั้น MLP หลัง attention"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """1 Transformer block = attention + feedforward (+ residual + norm)"""
    def __init__(self):
        super().__init__()
        self.sa = MultiHead(n_head, n_embd // n_head)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)      # ความหมายของ token
        self.pos_emb = nn.Embedding(block_size, n_embd)        # ตำแหน่ง
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)              # ทำนาย token ถัดไป

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]      # ตัดให้พอดี context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]            # เอาตำแหน่งสุดท้าย
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------- เทรน ----------
model = MiniGPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"โมเดลมี {n_params:,} พารามิเตอร์ | vocab={vocab_size} | device={device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(20)
        for k in range(20):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:4d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------- ลองให้มันสร้างข้อความ ----------
print("\n=== ตัวอย่างข้อความที่โมเดลสร้างเอง ===")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
