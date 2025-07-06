"""
Genera 100 pares (prompt1, prompt2) realistas.
80 en inglés, 20 en español. Guarda test_prompts.csv
"""

import random, csv, pathlib

EMOTIONS = [
    "overwhelmed", "lonely", "anxious", "frustrated", "sad",
    "calm", "nostalgic", "grateful", "hopeful", "motivated",
    "joyful", "confident"
]

TEMPLATES_EN = [
    "I'm feeling {emo1} after a rough day.",
    "Right now I'm {emo1}; could you guide me toward feeling {emo2}?",
    "Today was exhausting and I'm quite {emo1}. I want to end up {emo2}.",
    "I'm {emo1}. Help me reach a more {emo2} mood.",
    "Currently {emo1}. My goal is to feel {emo2} on my way home."
]

TEMPLATES_ES = [
    "Me siento {emo1} después de un día duro.",
    "Ahora mismo estoy {emo1}; ¿puedes llevarme a sentirme {emo2}?",
    "Ha sido agotador y estoy bastante {emo1}. Quiero terminar {emo2}.",
    "Estoy {emo1}. Ayúdame a llegar a un estado más {emo2}.",
    "Me siento {emo1}. Mi objetivo es sentirme {emo2} de camino a casa."
]

def make_pairs(n_pairs: int = 100, pct_en: float = 0.8):
    """
    Generates pairs of prompts based on emotion templates.
    --------
    Returns:
        list: A list of tuples containing prompt_1, prompt_2, emo_1, and emo_2.
    """
    pairs = []
    for _ in range(n_pairs):
        emo1, emo2 = random.sample(EMOTIONS, 2)
        if random.random() < pct_en:
            tpl = random.choice(TEMPLATES_EN)
        else:
            tpl = random.choice(TEMPLATES_ES)
        # Split into prompt1 / prompt2
        full = tpl.format(emo1=emo1, emo2=emo2)
        if ";" in full:
            p1, p2 = map(str.strip, full.split(";", 1))
        elif "." in full:
            first, rest = full.split(".", 1)
            p1 = first.strip() + "."
            p2 = rest.strip()
        else:
            p1 = full
            p2 = ""
        pairs.append((p1, p2, emo1, emo2))
    return pairs


def main():
    """
    Generates a CSV file containing test prompts and their associated emotions.
    --------
    Creates:
        test_prompts.csv: A CSV file with columns prompt_1, prompt_2, emo_1, and emo_2.
    """
    out = pathlib.Path("test_prompts.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_1", "prompt_2", "emo_1", "emo_2"])
        writer.writerows(make_pairs())
    print(f"✓ Saved {out.absolute()}")



if __name__ == "__main__":
    main()
