import os
import asyncio
import pandas as pd
from pathlib import Path
from emotions import detect_user_emotions, get_playlist_ids2_weighted
from context import get_context_embedding
from evaluation.playlist.eval_playlist import score_playlist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "evaluation" / "playlist" / "test_prompts.csv"
OUT_DIR = PROJECT_ROOT / "evaluation" / "playlist" / "output_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRACKS_MAX = 11  # o cargado desde env

async def run_weighted_eval():
    prompts = pd.read_csv(PROMPT_PATH).to_dict(orient="records")

    for w_emo in [round(x * 0.1, 1) for x in range(9, 0, -1)]:
        w_ctx = round(1 - w_emo, 1)
        results = []

        print(f"\nEvaluando para pesos emoción/contexto = {w_emo}/{w_ctx}")
        for row in prompts:
            p1, p2 = row["prompt_1"], row["prompt_2"]

            emb1, _ = detect_user_emotions(p1, n=3)
            emb2, _ = detect_user_emotions(p2, n=3)
            ctx1 = get_context_embedding(p1)
            ctx2 = get_context_embedding(p2)

            track_ids = get_playlist_ids2_weighted(
                emb1, emb2, ctx1, ctx2,
                genres=[], k=7, weight_emotion=w_emo, weight_context=w_ctx ,selection="best", n=1
            )[:N_TRACKS_MAX]

            result = await score_playlist(p1, p2, track_ids)
            results.append({
                **row,
                "track_ids": track_ids,
                **result["scores"],
                "overall": result["overall"],
                "rationale": result["rationale"],
            })
            print(f"✓ {p1[:30]}… → {result['overall']}")

        filename = f"weights_emo{int(w_emo*10)}_ctx{int(w_ctx*10)}.csv"
        out_path = OUT_DIR / filename
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"✓ Guardado: {out_path.name}")

if __name__ == "__main__":
    asyncio.run(run_weighted_eval())