from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from app.backend_factory import generate_reflection, generate_recommendation
from chatbots.GPT4omini import generate_translation, rewrite_as_emotion_statement
from emotions.emotions import detect_user_emotions
from context.context import get_context_embedding 
from selection.select_songs import get_playlist_ids_weighted
from urllib import request as urllib_request
import json
import asyncio
import os
from typing import List

BACKEND = os.getenv("BOT_BACKEND", "flan").lower()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, "static"),
            template_folder=os.path.join(BASE_DIR, "web"))
CORS(app)

def spotify_api_get(url: str, access_token: str) -> dict:
    """
    Does a GET request to the Spotify API (e.g. /v1/me),
    using urllib. Returns the decoded JSON as a dict.
    --------
    Args:
        url (str): The Spotify API endpoint URL.
        access_token (str): The OAuth access token for authentication.
    --------
    Returns:
        dict: The JSON response from the Spotify API, parsed as a dictionary.
    """
    req = urllib_request.Request(url)
    req.add_header("Authorization", f"Bearer {access_token}")
    with urllib_request.urlopen(req) as resp:
        if resp.status != 200:
            body = resp.read().decode()
            raise Exception(f"Spotify GET {url} falló: {resp.status} / {body}")
        data = resp.read().decode("utf-8")
        return json.loads(data)

def spotify_api_post(url: str, access_token: str, body_dict: dict) -> dict:
    """
    Does a POST to the Spotify API (e.g. /v1/users/{user_id}/playlists),
    using urllib. The body_dict is converted to JSON and sent as the request body.
    Returns the JSON response as a dict. Raises an exception if the request fails.
    --------
    Args:
        url (str): The Spotify API endpoint URL.
        access_token (str): The OAuth access token for authentication.
        body_dict (dict): The body of the POST request, to be sent as JSON.
    --------
    Returns:
        dict: The JSON response from the Spotify API, parsed as a dictionary.
    """
    body_bytes = json.dumps(body_dict).encode("utf-8")
    req = urllib_request.Request(url, data=body_bytes, method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    with urllib_request.urlopen(req) as resp:
        if resp.status not in (200, 201):
            body = resp.read().decode()
            raise Exception(f"Spotify POST {url} falló: {resp.status} / {body}")
        data = resp.read().decode("utf-8")
        return json.loads(data)

def get_playlist_link(spotify_token: str, song_list: List[str]) -> List[str]:
    """
    Creates a Spotify playlist with the given song IDs and returns the playlist URL.
    If no Spotify token is provided, returns the URLs of the individual tracks instead.
    --------
    Args:
        spotify_token (str): The OAuth access token for Spotify API.
        song_list (List[str]): List of Spotify track IDs to include in the playlist.
    --------
    Returns:
        List[str]: A list containing the URL of the created playlist or individual track URLs.
    """
    if not spotify_token:
        return [f"https://open.spotify.com/track/{tid}" for tid in song_list]
    
    me_data = spotify_api_get("https://api.spotify.com/v1/me", spotify_token)
    user_id = me_data.get("id")
    if not user_id:
        raise Exception("No se obtuvo user_id de Spotify.")

    # Creates a playlist
    create_body = {
        "name": "GPTune Your Emotions Playlist",
        "description": "Playlist generada automáticamente por GPTune Your Emotions",
        "public": False
    }
    create_url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    create_resp = spotify_api_post(create_url, spotify_token, create_body)

    playlist_id = create_resp.get("id")
    if not playlist_id:
        raise Exception("No se obtuvo playlist_id al crearla.")

    # Adds tracks
    uris = [f"spotify:track:{tid}" for tid in song_list]
    add_body = {"uris": uris}
    add_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    spotify_api_post(add_url, spotify_token, add_body)

    # Builds the final URL:
    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    return [playlist_url]

@app.route('/api/emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        input_number = data.get("input_number", 1)  # Input 1 or 2
        print(f"Original message (turn {input_number}): {message}")

        if BACKEND == "gpt4o-mini":
            translated_message = asyncio.run(generate_translation(message)) # Translate the message
            print(f"Translated message (turn {input_number}): {translated_message}")
            if input_number == 2:
                message2=translated_message
                translated_message = asyncio.run(rewrite_as_emotion_statement(message2)) # Reformulate the message
                print(f"Reformulated message (turn {input_number}): {translated_message}")
        else:
            translated_message = message

        emotional_embedding, emotions = detect_user_emotions(translated_message, n=3)

        if hasattr(emotional_embedding, "tolist"):
            embedding_list = emotional_embedding.tolist()
        else:
            embedding_list = emotional_embedding

        return jsonify({
            "emotion": emotions,
            "embedding": embedding_list
        }), 200

    except Exception as e:
        app.logger.exception("Error en /api/emotion:")
        return jsonify({
            "emotion": "sad",
            "embedding": [],
            "error": str(e)
        }), 200

@app.route('/api/reflect', methods=['POST'])
def reflect():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        response = generate_reflection(message)
        return jsonify({"response": response}), 200
    except Exception as e:
        app.logger.exception("Error en /api/reflect:")
        return jsonify({
            "response": "Lo siento, algo falló al generar la reflexión.",
            "error": str(e)
        }), 200

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data          = request.get_json(force=True) or {}
        ui1           = data.get("user_input_1", "")
        ui2           = data.get("user_input_2", "")
        br1           = data.get("bot_response_1", "")
        emotional_embedding_1          = data.get("emotional_embedding_1", [])
        emotional_embedding_2          = data.get("emotional_embedding_2", [])
        emo1         = data.get("emotion_detected_1", "")
        emo2         = data.get("emotion_detected_2", "")
        genres        = data.get("genres", []) # not being used because the dataset does not have genres, but kept for future use
        spotify_token = data.get("spotify_token")

        if BACKEND == "gpt4o-mini":
            reformulated_ui2 = asyncio.run(rewrite_as_emotion_statement(ui2))
        else:
            reformulated_ui2 = ui2
        
        context_embedding_1 = get_context_embedding(ui1)
        context_embedding_2 = get_context_embedding(reformulated_ui2)

        songs_ids = get_playlist_ids_weighted(emotional_embedding_1, emotional_embedding_2,
                                            context_embedding_1, context_embedding_2,
                                            weight_emotion=0.5, weight_context=0.5,
                                            k=7, selection='best', n=1)
    
        urls = get_playlist_link(spotify_token, songs_ids)
        response = generate_recommendation(ui1, ui2, br1, emo1, emo2)
        return jsonify({
            "response": response,
            "urls": urls
            }), 200
    except Exception as e:
        app.logger.exception("Error en /api/recommend:")
        return jsonify({
            "response": "No pude generar la recomendación.",
            "error": str(e)
        }), 200
    
@app.route('/')
def index():
    return render_template("GPTuneYourEmotions.html")
if __name__ == '__main__':
    app.run(debug=True)