#!/usr/bin/env python3
"""
Evalúa playlists usando GPT‑4 Turbo de Azure y guarda **un CSV por cada
combinación de pesos emoción / contexto**.

Salida del CSV (por fila):
    prompt_1,prompt_2,emo_1,emo_2,track_ids,
    emotional_alignment,progression,cohesion,diversity,
    overall_appeal,overall,rationale

Uso (ejemplo):
    python evaluation/playlist/playlist_evaluator.py \
           --prompts evaluation/playlist/test_prompts.csv \
           --out    results

Se generarán 9 archivos:
    results_emo01_ctx09.csv … results_emo09_ctx01.csv
Cada uno queda en la misma carpeta donde está el archivo --prompts.
"""
import os, csv, json, asyncio, pathlib, sys
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# ─────────────────────────── Paths & env ───────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOTENV_PATH  = PROJECT_ROOT / "data" / "credentials.env"
EMOTIONS_DIR = PROJECT_ROOT / "emotions"
CONTEXT_DIR  = PROJECT_ROOT / "context"

for directory in (EMOTIONS_DIR, CONTEXT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

load_dotenv(DOTENV_PATH)
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY      = os.getenv("AZURE_OPENAI_API_KEY", "")
DEPLOY_GPT4    = os.getenv("AZURE_GPT4_DEPLOYMENT", "")

from emotions import detect_user_emotions, get_playlist_ids2_weighted
# get_context_embedding puede no estar disponible en todos los entornos
try:
    from context import get_context_embedding  # type: ignore
except Exception:
    print("[WARN] get_context_embedding no encontrado; usando vector cero.")
    import numpy as np
    def get_context_embedding(_: str):  # type: ignore
        return np.zeros(768, dtype=float)

timeout_s = int(os.getenv("AZURE_TIMEOUT", "60"))
client = AsyncAzureOpenAI(api_version="2025-04-01-preview",
                          api_key=AZURE_KEY,
                          azure_endpoint=AZURE_ENDPOINT,
                          timeout=timeout_s)

PROMPT_FILE   = PROJECT_ROOT / "evaluation" / "playlist" / "eval_prompt.txt"
EVAL_PROMPT_SYS = PROMPT_FILE.read_text(encoding="utf-8")
N_TRACKS_MAX  = int(os.getenv("N_TRACKS_MAX", "11"))

# Pesos (emo, ctx) 0.1 – 0.9 … 0.9 – 0.1
WEIGHTS: List[Tuple[float, float]] = [
    (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6),
    (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)
]

# ─────────────────────────── Helpers ────────────────────────────

def build_eval_messages(p1: str, p2: str, track_ids: List[str]):
    """Crea el prompt de evaluación (sin Spotify API)."""
    playlist_lines = [f"{i}. https://open.spotify.com/track/{tid}"
                     for i, tid in enumerate(track_ids, 1)]
    return [
        {"role": "system", "content": EVAL_PROMPT_SYS},
        {"role": "user",
         "content": (
             f"Current emotion message: {p1}\n"
             f"Desired emotion message: {p2}\n"
             f"Playlist (order matters):\n" + "\n".join(playlist_lines)
         )},
    ]

async def score_playlist(p1: str, p2: str, track_ids: List[str]):
    rsp = await client.chat.completions.create(
        model=DEPLOY_GPT4,
        messages=build_eval_messages(p1, p2, track_ids),
        temperature=0.3,
        max_tokens=200,
    )
    return json.loads(rsp.choices[0].message.content)

async def evaluate_prompts(rows: list, w_emo: float, w_ctx: float) -> pd.DataFrame:
    """Devuelve DataFrame con las métricas para un par de pesos."""
    outputs = []
    for row in rows:
        p1, p2 = row["prompt_1"], row["prompt_2"]

        emb1, emo1 = detect_user_emotions(p1, n=1)
        emb2, emo2 = detect_user_emotions(p2, n=1)
        ctx1 = get_context_embedding(p1)
        ctx2 = get_context_embedding(p2)

        track_ids = get_playlist_ids2_weighted(
            emb1, emb2, ctx1, ctx2,
            weight_emotion=w_emo,
            weight_context=w_ctx,
            genres=[], k=7, selection="best", n=1,
        )[:N_TRACKS_MAX]

        scores = await score_playlist(p1, p2, track_ids)
        outputs.append({
            "prompt_1": p1,
            "prompt_2": p2,
            "emo_1":    emo1[0] if emo1 else "",
            "emo_2":    emo2[0] if emo2 else "",
            "track_ids": ";".join(track_ids),
            **scores["scores"],          # emotional_alignment,…,overall_appeal
            "overall":  scores["overall"],
            "rationale": scores["rationale"],
        })
        print(f"✓ {p1[:35]}… → {scores['overall']}")

    return pd.DataFrame(outputs)

# ─────────────────────────── Batch runner ─────────────────────────
async def batch_run(csv_path: str, out_prefix: str):
    """Evalúa todas las combinaciones de pesos y guarda 1 CSV por cada par."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_dir = pathlib.Path(csv_path).parent  # misma carpeta que los prompts

    for w_emo, w_ctx in WEIGHTS:
        print(f"\n— Evaluando w_emo={w_emo}  w_ctx={w_ctx} —")
        df = await evaluate_prompts(rows, w_emo, w_ctx)
        fname = f"{out_prefix}_emo{int(w_emo*10):02d}_ctx{int(w_ctx*10):02d}.csv"
        df.to_csv(out_dir / fname, index=False)
        print(f"  ↳ Guardado: {fname}")

# ─────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evalúa playlists y genera CSVs por peso.")
    parser.add_argument("--prompts", required=True,
                        help="CSV con columnas prompt_1,prompt_2…")
    parser.add_argument("--out", default="results",
                        help="Prefijo de los archivos de salida (sin extensión).")
    args = parser.parse_args()

    asyncio.run(batch_run(args.prompts, args.out))
