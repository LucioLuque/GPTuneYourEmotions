#!/usr/bin/env python3
"""
Evalúa la calidad de las playlists que genera tu recomendador
usando GPT-4 Turbo de Azure. NO depende de la API de Spotify.

Requisitos:
  • generate_test_prompts.py         (ya creado)
  • evaluation/recommender_wrapper.py (convierte prompts → track_ids)
  • eval_prompt.txt                  (rúbrica JSON)
  • data/credentials.env             con:
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_API_KEY
        AZURE_GPT4_DEPLOYMENT
"""

import os, json, csv, asyncio, pathlib, importlib, sys
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ──────────────── CONFIG ────────────────
load_dotenv("data/credentials.env")

AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_KEY      = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOY_GPT4    = os.environ["AZURE_GPT4_DEPLOYMENT"]

PROMPT_FILE    = pathlib.Path("/home/camila/Escritorio/GPTuneYourEmotions/evaluation/playlist/eval_prompt.txt")
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

# ───────────── Batch runner ─────────────
async def batch_run(csv_path: str, recommender_fn, out_path="results.parquet"):
    rows, outputs = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        p1, p2 = row["prompt_1"], row["prompt_2"]
        track_ids = recommender_fn(p1, p2)[:N_TRACKS_MAX]
        result = await score_playlist(p1, p2, track_ids)
        outputs.append({
            **row,
            "track_ids": track_ids,
            **result["scores"],
            "overall": result["overall"],
            "rationale": result["rationale"]
        })
        print(f"✓ {p1[:35]}… → {result['overall']}")

    pd.DataFrame(outputs).to_parquet(out_path)
    print(f"✓ Saved {out_path}")

# ────────────── CLI ──────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="/home/camila/Escritorio/GPTuneYourEmotions/evaluation/playlist/test_prompts.csv",
                        help="CSV con prompt_1, prompt_2…")
    parser.add_argument("--recommender",
                        default="evaluation.playlist.recommender_wrapper.recommend_from_prompts",
                        help="module.func que devuelve lista de IDs")
    parser.add_argument("--out", default="evaluation.playlist.results.parquet")
    args = parser.parse_args()

    mod, fn = args.recommender.rsplit(".", 1)
    recommender = getattr(importlib.import_module(mod), fn)

    asyncio.run(batch_run(args.batch, recommender, args.out))
