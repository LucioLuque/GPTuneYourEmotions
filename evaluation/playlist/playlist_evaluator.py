import os, json, csv, asyncio, pathlib, importlib, sys
from typing import List
import csv
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from emotions import detect_user_emotions
from selection import get_playlist_ids_weighted
from context import get_context_embedding 

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


def build_eval_messages(p1: str, p2: str, track_ids: List[str]) -> list:
    """
    Builds GPT messages for playlist evaluation using Spotify track links.
    --------
    Args:
        p1 (str): The current emotion message provided by the user.
        p2 (str): The desired emotion message specified by the user.
        track_ids (List[str]): List of track IDs to include in the playlist.
    --------
    Returns:
        list: A list of GPT messages formatted for playlist evaluation.
    """
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
    """
    Scores a playlist based on the given prompts and track IDs.
    --------
    Args:
        prompt_1 (str): The first user prompt.
        prompt_2 (str): The second user prompt.
        track_ids (list): List of track IDs to evaluate.
    --------
    Returns:
        dict: A dictionary containing scores, overall appeal, etc.
    """
    messages = build_eval_messages(p1, p2, track_ids)
    rsp = await client.chat.completions.create(
        model=DEPLOY_GPT4,
        messages=messages,
        temperature=0.3,
        max_tokens=200
    )
    return json.loads(rsp.choices[0].message.content)


async def batch_run(csv_path: str, output_dir="output_results"):
    """
    Processes prompts from a CSV file and generates playlists for each combination of weights.
    --------
    Args:
        csv_path (str): Path to the input CSV file containing prompts.
        output_dir (str): Directory to save the generated CSV files.
    --------
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    WEIGHTS = [ (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)]

    for emotion_weight, context_weight in WEIGHTS:
        outputs = []
        for row in rows:
            p1, p2 = row["prompt_1"], row["prompt_2"]

            # Obtain embeddings for the prompts
            emb1, _ = detect_user_emotions(p1, n=3)
            emb2, _ = detect_user_emotions(p2, n=3)
            context_embedding_1 = get_context_embedding(p1)
            context_embedding_2 = get_context_embedding(p2)

    
            track_ids = get_playlist_ids_weighted( emb1, emb2, context_embedding_1, context_embedding_2,
                genres=[], k=7, selection='best', n=1, weight_emotion=emotion_weight, 
                weight_context=context_weight)[:N_TRACKS_MAX]

            # Evaluate the playlist with GPT-4
            result = await score_playlist(p1, p2, track_ids)
            outputs.append({
                **row,
                "track_ids": track_ids,
                **result["scores"],
                "overall": result["overall"],
                "rationale": result["rationale"]
            })
            print(f"{p1[:35]}… → {result['overall']}")

        # Save results to CSV
        filename = f"weights_{emotion_weight}_{context_weight}.csv"
        output_path = os.path.join(output_dir, filename)
        pd.DataFrame(outputs).to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="/home/bryan/Documentos/GitHub/GPTuneYourEmotions/evaluation/playlist/test_prompts.csv",
                        help="CSV con prompt_1, prompt_2…")
    parser.add_argument("--out", default="evaluation.playlist.results.csv")
    args = parser.parse_args()

    asyncio.run(batch_run(args.batch, args.out))