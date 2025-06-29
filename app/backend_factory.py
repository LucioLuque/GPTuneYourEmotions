import os, asyncio
from emotions.emotions import detect_user_emotions  

BACKEND = os.getenv("BOT_BACKEND", "flan").lower()   # flan | gpt4o-mini
print(f"[backend_factory] BOT_BACKEND = '{BACKEND}'")  # imprime "flan" o "gpt4o-mini"
#Importa las funciones de flan y de gpt-4o-mini y les pone un alias
from chatbots.flan_t5 import (
    generate_reflection   as _flan_reflect,
    generate_recommendation as _flan_recommend,
)

from chatbots.GPT4omini import (
    generate_reflection     as _g4o_reflect_async,
    generate_recommendation as _g4o_recommend_async,
)

if BACKEND == "gpt4o-mini":
    def generate_reflection(message: str):
        """
        1) Detecta la emoción dominante con tu clasificador local.
        2) Llama al GPT-4o (corrutina) y devuelve la respuesta final.
        """
        _, emos = detect_user_emotions(message, n=1)
        detected = emos[0] if emos else "neutral"
        return asyncio.run(_g4o_reflect_async(message, detected))

    def generate_recommendation(ui1, ui2, br1, emo1, emo2, song):
        """
        Adapta los parametros (5 strings) al formato que espera GPT-4o.
        """
        history = [
            {"role": "user",      "content": ui1},
            {"role": "assistant", "content": br1},
            {"role": "user",      "content": ui2},
        ]
        return asyncio.run(_g4o_recommend_async(
            history,
            song_title=song,
            current_emotion=emo1,
            desired_emotion=emo2,
        ))

else:
    # ---------- Flan-passthrough ------------------------------------------
    def generate_reflection(message: str):
        return _flan_reflect(message)

    def generate_recommendation(ui1, ui2, br1, emo1, emo2, song):
        return _flan_recommend(ui1, ui2, br1, emo1, emo2, song)
