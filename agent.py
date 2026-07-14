import os
import sys
import time
import torch
from model import MiniGPT

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# Load checkpoint
# =========================
if not os.path.exists("checkpoints/mini_llm.pt"):
    print("Error: checkpoints/mini_llm.pt not found. Run the training script first.")
    exit(1)

checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]
vocab_size = checkpoint["vocab_size"]
config = checkpoint["config"]

# =========================
# Encode / Decode
# =========================
def encode(s: str):
    return [stoi[c] for c in s if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

# =========================
# Load model
# =========================
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

# =========================
# Settings
# =========================
MAX_NEW_TOKENS = 250  
TEMPERATURE = 0.01  # FIX: Set practically to 0 to force exact text lookup instead of guessing words
MAX_HISTORY_CHARS = 1000
TYPE_DELAY = 0.015  

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    text = text.replace("\r", "").strip()
    return text if text else "..."

def type_out(text: str, delay: float = TYPE_DELAY):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def save_chat_log(chat_history: str, filename: str = "chat_log.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(chat_history)

def build_model_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    known_prefixes = (
        "topic:",
        "question:",
        "answer:",
        "problem:",
        "explanation:",
        "solution:",
        "analysis:",
        "concept:",
    )
    question_starters = ("what ", "why ", "how ", "when ", "where ", "who ", "which ")

    if lowered.startswith(known_prefixes):
        return prompt

    if prompt.endswith("?") or lowered.startswith(question_starters):
        return f"Question: {prompt}\nAnswer:"

    return f"Topic: {prompt}\n"

def generate_response(prompt: str, history: str = "") -> str:
    full_prompt = build_model_prompt(prompt)

    encoded = encode(full_prompt)
    if not encoded:
        return "I cannot understand this prompt because the characters were not in the training data."

    context = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )[0].tolist()

    full_output = decode(generated)
    
    # Extract only the newly generated text
    if full_output.startswith(full_prompt):
        reply = full_output[len(full_prompt):].strip()
    else:
        reply = full_output.strip()
    
    # Stop generation cleanly if it loops back into another dataset block
    for separator in ["\nTopic:", "\nQuestion:", "\nAnswer:"]:
        sep_pos = reply.find(separator)
        if sep_pos != -1:
            reply = reply[:sep_pos].strip()

    return clean_text(reply)

# =========================
# Intro
# =========================
print("\nMini LLM Agent")
print("Commands:")
print("  /exit   - quit")
print("  /clear  - clear memory")
print("  /save   - save chat")
print("  /fast   - faster typing")
print("  /slow   - slower typing")
print("  /temp X - set temperature, example: /temp 0.8\n")

chat_history = ""

# =========================
# Chat loop
# =========================
while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "/exit":
        print("Agent: Goodbye.")
        break

    if user_input.lower() == "/clear":
        chat_history = ""
        print("Agent: Memory cleared.")
        continue

    if user_input.lower() == "/save":
        save_chat_log(chat_history)
        print("Agent: Chat saved to chat_log.txt")
        continue

    if user_input.lower() == "/fast":
        TYPE_DELAY = 0.005
        print("Agent: Typing speed set to fast.")
        continue

    if user_input.lower() == "/slow":
        TYPE_DELAY = 0.03
        print("Agent: Typing speed set to slow.")
        continue

    if user_input.lower().startswith("/temp "):
        try:
            value = float(user_input.split(" ", 1)[1])
            if value < 0:
                print("Agent: Temperature cannot be negative.")
                continue
            TEMPERATURE = max(0.01, value)  # Prevent complete 0 split errors
            print(f"Agent: Temperature set to {TEMPERATURE}")
        except ValueError:
            print("Agent: Invalid temperature value.")
        continue

    response = generate_response(user_input, history=chat_history)

    print("Agent: ", end="", flush=True)
    type_out(response, TYPE_DELAY)
    print()

    chat_history += f"User: {user_input}\nAgent: {response}\n"

    if len(chat_history) > MAX_HISTORY_CHARS:
        chat_history = chat_history[-MAX_HISTORY_CHARS:]