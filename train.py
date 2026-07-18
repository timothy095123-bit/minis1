import os
import torch
import time
from model import MiniGPT

# =========================
# Hyperparameters
# =========================
device = "cpu" # Optimized for Core i3 CPU performance

# Updated training configuration
batch_size = 8
block_size = 96
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 10

n_embd = 96
n_head = 4
n_layer = 3
dropout = 0.1
print(f"Using device: {device} (i3 Lightning Mode)")

torch.manual_seed(1337)

# =========================
# Load text
# =========================
if not os.path.exists("data/train.txt"):
    print("Error: data/train.txt not found. Please create the file first.")
    exit(1)

with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# Build vocabulary
# =========================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

train_data = data
val_data = data 

def get_batch(split):
    source = train_data if split == "train" else val_data
    max_idx = max(1, len(source) - block_size)
    ix = torch.randint(0, max_idx, (batch_size,))
    
    x_list, y_list = [], []
    for i in ix:
        idx = i.item()
        chunk_x = source[idx : idx + block_size]
        chunk_y = source[idx + 1 : idx + block_size + 1]
        
        # Safe dataset boundary padding logic
        if len(chunk_x) < block_size:
            padding = torch.zeros(block_size - len(chunk_x), dtype=torch.long)
            chunk_x = torch.cat([chunk_x, padding])
        if len(chunk_y) < block_size:
            padding = torch.zeros(block_size - len(chunk_y), dtype=torch.long)
            chunk_y = torch.cat([chunk_y, padding])
            
        x_list.append(chunk_x)
        y_list.append(chunk_y)
        
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =========================
# Create model
# =========================
model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =========================
# Train loop
# =========================
print(f"\nStarting training for {max_iters} iterations...")
# Parameters dropped from 0.85M to an ultra-lean 0.12M for speed
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

start_time = time.time()
for step in range(max_iters):
    if step > 0 and (step % eval_interval == 0 or step == max_iters - 1):
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        print(f"step {step:4d} | train loss {losses['train']:.4f} | time {elapsed:.1f}s")

    elif step % 100 == 0:
        elapsed = time.time() - start_time
        print(f"step {step:4d} | training... | time {elapsed:.1f}s")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nTraining completed in {time.time() - start_time:.1f}s")

# =========================
# Save checkpoint
# =========================
os.makedirs("checkpoints", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "config": {
        "n_embd": n_embd,
        "block_size": block_size,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
}

torch.save(checkpoint, "checkpoints/mini_llm.pt")
print("Saved checkpoint to checkpoints/mini_llm.pt")