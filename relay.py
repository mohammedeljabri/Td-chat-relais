"""
Relais minimal pour le chat du TD (Moodle).
La cle API reste ICI, cote serveur. Le navigateur des etudiants n'en voit jamais.

Lancer en local :
    pip install -r requirements.txt
    export LLM_API_KEY="votre_cle"        # (Windows PowerShell : $env:LLM_API_KEY="...")
    uvicorn relay:app --port 8000 --reload

Le chat de la page appellera alors  http://localhost:8000/chat
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("LLM_API_KEY", "")

# Fournisseur actif : Mistral (vous l'avez deja fonctionnel)
PROVIDER_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-latest"
# Pour Groq, commenter les 2 lignes ci-dessus et decommenter :
# PROVIDER_URL = "https://api.groq.com/openai/v1/chat/completions"
# MODEL = "llama-3.3-70b-versatile"

# Origines autorisees a appeler ce relais.
# En PRODUCTION, remplacez "*" par l'URL exacte de votre Moodle,
# ex : ["https://moodle.mon-etablissement.fr"]
ALLOWED_ORIGINS = ["*"]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Relais chat TD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    messages: list  # [{role, content}, ...] envoye par la page


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
async def chat(body: ChatIn):
    if not API_KEY:
        raise HTTPException(500, "Cle API non configuree (variable LLM_API_KEY).")

    payload = {
        "model": MODEL,
        "messages": body.messages,
        "temperature": 0.3,
        "max_tokens": 700,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(PROVIDER_URL, json=payload, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(502, f"Fournisseur injoignable : {e}")

    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(502, "Reponse du fournisseur inattendue.")

    return {"reply": content}
