import os
import asyncio
from typing import List, Dict

from openai import AsyncAzureOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai import RateLimitError, APITimeoutError, APIConnectionError

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
        " • Respond that you will give the user a playlist that will help them transition from the current emotion to the desired emotion.\n"
        " • Write that you can find the playlist link below with ':'.\n"
        " • You MUST NOT add a link.\n"
    )},
    {"role": "user", "content": "Everything feels out of control lately."},
    {"role": "assistant",
     "content": (
        "I'm really sorry you're feeling overwhelmed. Remember that you're not "
        "alone and it's okay to take things one step at a time. "
        "I hope these songs help you feel better and support you in the process. "
        "You can find the link to the Spotify playlist here:"
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
             #f"Song to recommend: {song_title}"
             #f"IMPORTANT INSTRUCTION: You MUST recommend EXACTLY the song '{song_title}' - "
             f"Do not recommend any other song or artist, the link to the song is already provided. "
             f"do not substitute it with any other song or similar artist. "
             f"The songs selection is final and has been carefully chosen for this emotional transition. "
             #f"Explain why this specific song can help with the transition from {current_emotion} to {desired_emotion}."
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



async def generate_translation(user_message: str) -> str:
    prompt = (
        "Please translate the following user message to English, preserving especially the emotional meaning:\n\n"
        f"{user_message}\n\n"
        "Translate literally, but also ensure that the emotional tone and nuances are maintained."
        "Respond ONLY with the translated text. Do not add explanations."
    )

    messages = [{"role": "user", "content": prompt}]
    response = await safe_chat_create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


async def rewrite_as_emotion_statement(user_input: str) -> str:
    """
    Rewrites user input as a direct emotional statement.
    Example: 'I want to feel happy' → 'I feel happy'
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that reformulates user messages into direct emotional statements, preserving the intended emotion and important context if available.\n\n"
                "• If the user says something like 'I want to feel happy because tomorrow is my wedding', you must rewrite it as: 'I feel happy because tomorrow is my wedding'.\n"
                "• The rewritten sentence must be declarative and use present tense (e.g., 'feel ___'), avoiding desires like 'want', 'wish', or 'would like'.\n"
                "• If there are multiple emotions, include all of them.\n"
                "• Never omit important context like reasons, events, or explanations that the user provided.\n"
                "• Do not introduce new information. Just rewrite faithfully.\n"
                "• Avoid ambiguous statements that could be interpreted as expressing a desire unless explicitly stated.\n"
                "• Output only the rewritten sentence."
            )
        },
        {
            "role": "user",
            "content": f"Original message: {user_input}\nRewritten:"
        }
    ]

    response = await safe_chat_create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=0.3,
        max_tokens=30,
        top_p=0.9,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    return response.choices[0].message.content.strip()
