import torch
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
config = checkpoint["config"]

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    n_head=config["n_head"],
    n_layer=config["n_layer"],
    dropout=config["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = input("Enter prompt: ").strip()

if prompt:
    encoded = encode(prompt)
    if not encoded:
        print("Prompt has no known characters from training data.")
        raise SystemExit
    context = torch.tensor([encoded], dtype=torch.long, device=device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

generated = model.generate(
    context,
    max_new_tokens=300,
    temperature=0.9
)[0].tolist()

print("\n=== GENERATED TEXT ===\n")
print(decode(generated))