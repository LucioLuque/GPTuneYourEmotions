#!/usr/bin/env python3
"""
Evalúa la calidad de las playlists que genera tu recomendador
usando GPT-4 Turbo de Azure. NO depende de la API de Spotify.

Requisitos:
  • generate_test_prompts.py         (ya creado)
  • evaluation/recommender_wrapper.py (convierte prompts → track_ids)
  • eval_prompt.txt                  (rúbrica JSON)
  • data/credentials.env             con:
        AZURE_ENDPOINT
        AZURE_KEY
        DEPLOY_GPT4
"""

import os, json, csv, asyncio, pathlib, importlib, sys
from typing import List


import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / "data" / "credentials.env"
EMOTIONS_DIR = PROJECT_ROOT / "emotions"
CONTEXT_DIR = PROJECT_ROOT / "context"

for directory in (EMOTIONS_DIR, CONTEXT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

load_dotenv(DOTENV_PATH)

AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_KEY      = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOY_GPT4    = os.environ["AZURE_GPT4_DEPLOYMENT"]

from emotions import detect_user_emotions, get_playlist_ids2_weighted
from context import get_context_embedding 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


PROMPT_FILE = PROJECT_ROOT / "evaluation" / "playlist" / "eval_prompt.txt"

N_TRACKS_MAX   = int(os.getenv("N_TRACKS_MAX", 11))

client = AsyncAzureOpenAI(
    api_version="2025-04-01-preview",
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    timeout=60
)

EVAL_PROMPT_SYS = PROMPT_FILE.read_text()

# ────────── Build GPT messages (sin Spotify) ──────────
def build_eval_messages(p1: str, p2: str, track_ids: List[str]) -> list:
    playlist_lines = [
        f"{i}. https://open.spotify.com/track/{tid}"
        for i, tid in enumerate(track_ids, 1)
    ]
    playlist_block = "\n".join(playlist_lines)
    return [
        {"role": "system", "content": EVAL_PROMPT_SYS},
        {"role": "user",
         "content": f"Current emotion message: {p1}\n"
                    f"Desired emotion message: {p2}\n"
                    f"Playlist (order matters):\n{playlist_block}"}
    ]

async def score_playlist(p1: str, p2: str, track_ids: List[str]) -> dict:
    messages = build_eval_messages(p1, p2, track_ids)
    rsp = await client.chat.completions.create(
        model=DEPLOY_GPT4,
        messages=messages,
        temperature=0.3,
        max_tokens=200
    )
    return json.loads(rsp.choices[0].message.content)

async def evaluate_playlists(prompts: list, weight_emotion=0.4, weight_context=0.6, n_tracks_max=11) -> pd.DataFrame:
    """
    Evalúa las playlists generadas a partir de una lista de prompts.
    
    Args:
        prompts (list): Lista de diccionarios con "prompt_1" y "prompt_2".
        weight_emotion (float): Peso para los embeddings emocionales.
        weight_context (float): Peso para los embeddings contextuales.
        n_tracks_max (int): Número máximo de canciones en la playlist.

    Returns:
        pd.DataFrame: Resultados de la evaluación.
    """
    outputs = []
    for row in prompts:
        p1, p2 = row["prompt_1"], row["prompt_2"]

        # Generar embeddings emocionales y contextuales
        emb1, _ = detect_user_emotions(p1, n=3)
        emb2, _ = detect_user_emotions(p2, n=3)
        context_embedding_1 = get_context_embedding(p1)
        context_embedding_2 = get_context_embedding(p2)

        # Generar IDs de la playlist con pesos personalizados
        track_ids = get_playlist_ids2_weighted(
            emb1, emb2,
            context_embedding_1, context_embedding_2,
            genres=[],              # sin filtros
            k=7, selection='best', n=1
        )[:n_tracks_max]

        # Evaluar la playlist con GPT-4
        result = await score_playlist(p1, p2, track_ids)
        outputs.append({
            **row,
            "track_ids": track_ids,
            **result["scores"],
            "overall": result["overall"],
            "rationale": result["rationale"]
        })
        print(f"✓ {p1[:35]}… → {result['overall']}")

    return pd.DataFrame(outputs)



# ───────────── Batch runner ─────────────
import os
import csv
import pandas as pd

# ───────────── Batch runner ─────────────
async def batch_run(csv_path: str, output_dir="output_results"):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Leer las filas del archivo CSV
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    rows = rows[:20]  # Tomar el primer décimo de las filas

    # Lista de pesos para emoción y contexto
    WEIGHTS = [
        (0.5, 0.5),
    ]

    # Iterar sobre cada combinación de pesos
    for emotion_weight, context_weight in WEIGHTS:
        outputs = []
        for row in rows:
            p1, p2 = row["prompt_1"], row["prompt_2"]

            # Obtener embeddings y contexto
            emb1, _ = detect_user_emotions(p1, n=3)
            emb2, _ = detect_user_emotions(p2, n=3)
            context_embedding_1 = get_context_embedding(p1)
            context_embedding_2 = get_context_embedding(p2)

            # Generar IDs de pistas con los pesos actuales
            track_ids = get_playlist_ids2_weighted(
                emb1, emb2,
                context_embedding_1, context_embedding_2,
                genres=[],              # sin filtros
                k=7, selection='best', n=1,  # → 11 canciones
                weight_emotion=emotion_weight, weight_context=context_weight
            )[:N_TRACKS_MAX]

            # Evaluar la playlist
            result = await score_playlist(p1, p2, track_ids)
            outputs.append({
                **row,
                "track_ids": track_ids,
                **result["scores"],
                "overall": result["overall"],
                "rationale": result["rationale"]
            })
            print(f"✓ {p1[:35]}… → {result['overall']}")

        # Guardar resultados como CSV
        filename = f"weights_{emotion_weight}_{context_weight}.csv"
        output_path = os.path.join(output_dir, filename)
        pd.DataFrame(outputs).to_csv(output_path, index=False)
        print(f"✓ Saved {output_path}")

# ────────────── CLI ──────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="/home/bryan/Documentos/GitHub/GPTuneYourEmotions/evaluation/playlist/test_prompts.csv",
                        help="CSV con prompt_1, prompt_2…")
    parser.add_argument("--out", default="evaluation.playlist.results.csv")
    args = parser.parse_args()

    asyncio.run(batch_run(args.batch, args.out))

# ────────────── CLI ──────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="/home/bryan/Documentos/GitHub/GPTuneYourEmotions/evaluation/playlist/test_prompts.csv",
                        help="CSV con prompt_1, prompt_2…")
    parser.add_argument("--out", default="evaluation.playlist.results.csv")
    args = parser.parse_args()


    asyncio.run(batch_run(args.batch, args.out))
