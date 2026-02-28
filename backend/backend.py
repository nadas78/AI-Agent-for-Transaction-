# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# =========================
# 1️⃣ Configuration modèle
# =========================
# Remplacez par votre repo Hugging Face, exemple : "username/defi-chatbot"
repo_id = "username/defi-chatbot"

print("🔹 Chargement du modèle depuis Hugging Face...")
base_model = AutoModelForCausalLM.from_pretrained(repo_id)
model = PeftModel.from_pretrained(base_model, repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

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
    allow_origins=["*"],  # remplacer "*" par le frontend autorisé en prod
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
    # Prompt identique au fine-tuning
    prompt = (
        f"{query.text}\n"
        "Provide technical explanation, simplified explanation, severity, recommended fix and confidence:"
    )

    # Tokenization
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Génération
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=query.max_length,
            top_k=query.top_k,
            temperature=query.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Décodage
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Supprimer le prompt initial si présent
    if output_text.startswith(query.text):
        output_text = output_text[len(query.text):].strip()

    return {"input": query.text, "output": output_text}

# =========================
# 5️⃣ Run avec uvicorn
# =========================
# Pour exécuter : uvicorn backend:app --host 0.0.0.0 --port 8000 --reload