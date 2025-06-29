import json, pathlib
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOTENV_PATH  = PROJECT_ROOT / "data" / "credentials.env"
load_dotenv(DOTENV_PATH)

AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_KEY      = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOY_GPT4    = os.environ["AZURE_GPT4_DEPLOYMENT"]

EVAL_SYS = (
    "You are a senior music-therapy expert. "
    "Judge how well a song matches a user's emotional and contextual prompt.\n"
    "Use the following 5-criterion rubric, scoring each from 0 (very poor) to 5 (excellent):\n"
    "1) Emotional alignment: how well the song reflects the emotion of the prompt.\n"
    "2) Contextual relevance: how well the song fits the theme or context.\n"
    "3) Lyrical coherence: the fluency and sense of the lyrics.\n"
    "Return a JSON with "
    "{\"scores\": {\"emotional\": X, \"contextual\": Y, "
    "\"coherence\": Z}, \"overall\": O, \"rationale\": \"…\"}."
)

client = AsyncAzureOpenAI(
    api_version="2025-04-01-preview",
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    timeout=60
)

def build_song_messages(prompt: str, song: str) -> list:
    return [
        {"role": "system", "content": EVAL_SYS},
        {"role": "user", "content":
            f"User prompt: {prompt}\n"
            f"Recommended song lyrics:\n{song}"
        }
    ]

async def score_song(prompt: str, song: str) -> dict:
    msgs = build_song_messages(prompt, song)
    try:
        rsp = await client.chat.completions.create(
            model=DEPLOY_GPT4,
            messages=msgs,
            temperature=0.2,
            max_tokens=300
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        # Solo para BadRequestError con content_filter
        if "content_filter" in str(e):
            return {
                "scores": {"emotional": None, "contextual": None, "coherence": None},
                "overall": None,
                "rationale": "The song was filtered out by the content filter."
            }
        else:
            raise


async def run_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        prompt = row["text"]
        song    = row["lyrics_predictions"]
        result = await score_song(prompt, song)
        records.append({
            **row.to_dict(),
            **result["scores"],
            "overall": result["overall"],
            "rationale": result["rationale"]
        })
    return pd.DataFrame(records)