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
MAX_NEW_TOKENS = 300  # Allow longer, more complete responses
TEMPERATURE = 0.8  # Balanced - creative but still coherent
MAX_HISTORY_CHARS = 1000
TYPE_DELAY = 0.015  # mas mababa = mas mabilis mag-type

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    text = text.replace("\r", "").strip()
    return text if text else "..."

def type_out(text: str, delay: float = TYPE_DELAY):
    """
    Simulated streaming / typing effect.
    """
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def save_chat_log(chat_history: str, filename: str = "chat_log.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(chat_history)

def generate_response(prompt: str, history: str = "") -> str:
    """
    Generate text based on the prompt.
    Since the model was trained on plain text, we use the prompt directly.
    """
    # Use just the prompt, not conversation format
    full_prompt = prompt

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
    
    # Remove the original prompt from the output
    if full_output.startswith(prompt):
        reply = full_output[len(prompt):].strip()
    else:
        reply = full_output.strip()
    
    # Better stopping: stop at double newline or sentence end
    # This preserves single newlines in code blocks but stops at paragraph breaks
    stop_markers = ['\n\n', '. ', '? ', '! ']
    min_stop = len(reply)
    
    for marker in stop_markers:
        pos = reply.find(marker)
        if pos != -1 and pos < min_stop:
            min_stop = pos + len(marker.rstrip())
    
    if min_stop < len(reply):
        reply = reply[:min_stop].strip()
    
    # Limit to reasonable length (first 500 chars max)
    if len(reply) > 500:
        # Try to stop at sentence boundary
        for i in range(500, 200, -1):
            if reply[i] in '.!?':
                reply = reply[:i+1]
                break
        else:
            reply = reply[:500] + "..."

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
            if value <= 0:
                print("Agent: Temperature must be greater than 0.")
                continue
            TEMPERATURE = value
            print(f"Agent: Temperature set to {TEMPERATURE}")
        except ValueError:
            print("Agent: Invalid temperature value.")
        continue

    response = generate_response(user_input, history=chat_history)

    print("Agent: ", end="", flush=True)
    type_out(response, TYPE_DELAY)
    print()

    # Store simpler history - just for logging, not used in generation
    chat_history += f"User: {user_input}\nAgent: {response}\n"

    if len(chat_history) > MAX_HISTORY_CHARS:
        chat_history = chat_history[-MAX_HISTORY_CHARS:]