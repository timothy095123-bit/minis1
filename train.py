import os
import torch
import time
# Assuming your MiniGPT model is imported from model.py
from model import MiniGPT

# =========================
# Hyperparameters
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU-optimized settings tailored for small custom Q&A memorization
if device == "cpu":
    batch_size = 8  
    block_size = 64  # Lowered slightly to match short Q&A line limits better
    max_iters = 6000  # Doubled iterations so the model completely overfits/memorizes the text
    eval_interval = 500  
    learning_rate = 5e-4  # Slightly higher learning rate for fast memorization
    eval_iters = 10  
    
    n_embd = 96  
    n_head = 4
    n_layer = 3
    dropout = 0.0  # Removed dropout so it memorizes perfectly instead of generalizing
    print(f"Using device: {device} (CPU Overfit Mode)")
else:
    # GPU settings
    batch_size = 32
    block_size = 64
    max_iters = 6000
    eval_interval = 500  
    learning_rate = 5e-4
    eval_iters = 50
    
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.0
    print(f"Using device: {device} (GPU Overfit Mode)")

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

# Small Dataset Fix: Use 100% of data for training to ensure perfect memorization
train_data = data
val_data = data 

def get_batch(split):
    source = train_data if split == "train" else val_data
    # Prevent index out of bounds if text is extremely short
    max_idx = max(1, len(source) - block_size)
    ix = torch.randint(0, max_idx, (batch_size,))
    
    x_list, y_list = [], []
    for i in ix:
        # Pad with 0 (or space character) if text segment is smaller than block_size
        chunk_x = source[i:i + block_size]
        chunk_y = source[i + 1:i + block_size + 1]
        
        if len(chunk_x) < block_size:
            chunk_x = torch.cat([chunk_x, torch.zeros(block_size - len(chunk_x), dtype=torch.long)])
        if len(chunk_y) < block_size:
            chunk_y = torch.cat([chunk_y, torch.zeros(block_size - len(chunk_y), dtype=torch.long)])
            
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
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

start_time = time.time()
for step in range(max_iters):
    if step > 0 and (step % eval_interval == 0 or step == max_iters - 1):
        eval_start = time.time()
        losses = estimate_loss(model)
        eval_time = time.time() - eval_start
        elapsed = time.time() - start_time
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | time {elapsed:.1f}s")

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