import os
import asyncio
from typing import List, Dict

from openai import AsyncAzureOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai import RateLimitError, APITimeoutError, APIConnectionError

#### Se pasa por terminal
ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY    = os.getenv("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini-tutorial")
API_VERSION = "2025-04-01-preview"         

client = AsyncAzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    timeout=30,               # segundos
    max_retries=3             # reintentos automáticos
)

# ───────────────────────────────────
# Seguridad
# ───────────────────────────────────
async def safe_chat_create(**kwargs):
    """
    Envuelve la llamada con reintentos exponenciales
    para errores previsibles.
    """
    backoff = 1
    for attempt in range(5):
        try:
            return await client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == 4:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2

FEW_SHOT_REFLECTION = [
    {"role": "system",
     "content": (
        "You are an emotionally intelligent assistant. "
        "For the first user turn you must:\n"
        " • acknowledge the detected emotion\n"
        " • respond in at least two supportive sentences\n"
        " • finish by asking: 'How would you like to feel instead?'\n"
        "Do NOT recommend a song in this turn."
    )},
    {"role": "user", "content": "I feel like nothing is going right."},
    {"role": "assistant",
     "content": (
        "I'm sorry you're feeling that way. It can be discouraging when things "
        "don't work out as planned. How would you like to feel instead?"
     )},
]

FEW_SHOT_RECOMMEND = [
    {"role": "system",
     "content": (
        "You are an emotionally intelligent assistant. "
        "In this turn you must:\n"
        " • acknowledge the user's current emotion and desired emotion\n"
        " • respond in at least three supportive sentences\n"
        " • recommend ONE song in a meaningful way and explain why it fits."
        # PARA MI DEBERIA TERMINAR DICIENDO 'you can listen the playlist here:' o algo asi
    )},
    {"role": "user", "content": "Everything feels out of control lately."},
    {"role": "assistant",
     "content": (
        "I'm really sorry you're feeling overwhelmed. Remember that you're not "
        "alone and it's okay to take things one step at a time. "
        "A song that might help is 'Fix You' by Coldplay; its gentle build and "
        "hopeful lyrics can offer comfort and remind you that things can improve."
     )},
]


async def generate_reflection(user_message: str, detected_emotion: str) -> str:
    """
    Primera interacción: dar contención y preguntar
    cómo le gustaría sentirse.
    """
    messages = FEW_SHOT_REFLECTION + [
        {"role": "user", "content": user_message},
        {"role": "system",
         "content": (f"Detected emotion: {detected_emotion}"
                    f"Give a supportive response and ask how the user would like to feel instead."
                    f"Be naturally empathetic and friendly, as if you were a human friend. " 
        )}                   
    ]

    response = await safe_chat_create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.7,
        max_tokens=200,
        top_p=0.9,
        presence_penalty=0.6,
    )
    return response.choices[0].message.content.strip()

async def generate_recommendation(
    history: list,
    song_title: str,
    current_emotion: str,
    desired_emotion: str
) -> str:
    """
    Segunda interacción: recomendar canción.
    'history' debe contener la conversación real previa (incl. system).
    """
    messages = FEW_SHOT_RECOMMEND + history + [
        {"role": "system",
         "content": (
             f"Current emotion: {current_emotion}\n"
             f"Desired emotion: {desired_emotion}\n"
             f"Song to recommend: {song_title}"
             f"IMPORTANT INSTRUCTION: You MUST recommend EXACTLY the song '{song_title}' - "
             f"do not substitute it with any other song or similar artist. "
             f"The song selection is final and has been carefully chosen for this emotional transition. "
             f"Explain why this specific song can help with the transition from {current_emotion} to {desired_emotion}."
        )}
    ]
    response = await safe_chat_create(
        model=DEPLOYMENT,
        messages= messages,
        temperature=0.7,
        max_tokens=250,
        top_p=0.9,
        presence_penalty=0.6,
    )
    return response.choices[0].message.content.strip()

