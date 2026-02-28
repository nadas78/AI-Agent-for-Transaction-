# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# =========================
# 1️⃣ Configuration modèle
# =========================

MODEL_PATH = "./exported_model"  # ✅ ton dossier local

print("🔹 Chargement du modèle local...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"✅ Modèle chargé sur {device}")

# =========================
# 2️⃣ FastAPI + CORS
# =========================
app = FastAPI(title="DeFi AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 3️⃣ Schema de requête
# =========================
class Query(BaseModel):
    text: str
    max_length: int = 256
    top_k: int = 50
    temperature: float = 0.7

# =========================
# 4️⃣ Endpoint /predict
# =========================
@app.post("/predict")
def predict(query: Query):
    prompt = (
        f"{query.text}\n"
        "Provide technical explanation, simplified explanation, severity, recommended fix and confidence:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=query.max_length,
            top_k=query.top_k,
            temperature=query.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 🔹 nettoyage du prompt
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].strip()

    return {
        "input": query.text,
        "output": output_text
    }

# =========================
# 5️⃣ Run avec uvicorn
# =========================
# uvicorn backend:app --host 0.0.0.0 --port 8000 --reload