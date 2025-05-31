from flask import Flask, request, jsonify
from flask_cors import CORS
from backend_factory import generate_reflection, generate_recommendation
from emotions import detect_user_emotions, get_playlist_ids
from urllib import request as urllib_request
import json

app = Flask(__name__)
CORS(app)

def spotify_api_get(url: str, access_token: str) -> dict:
    """
    Hace un GET a la API de Spotify (p.ej. /v1/me), usando urllib.
    Retorna el JSON decodificado como dict. Lanza excepción si falló.
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
    Hace un POST a la API de Spotify con un cuerpo JSON (body_dict) usando urllib.
    Retorna el JSON decodificado como dict. Lanza excepción si no es 200/201.
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
    
def get_playlist_link(spotify_token, song_list):
    if not spotify_token:
        return [f"https://open.spotify.com/track/{tid}" for tid in song_list]
    
    me_data = spotify_api_get("https://api.spotify.com/v1/me", spotify_token)
    user_id = me_data.get("id")
    if not user_id:
        raise Exception("No se obtuvo user_id de Spotify.")

    # Crear la playlist (privada)
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

    # Agregar tracks
    uris = [f"spotify:track:{tid}" for tid in song_list]
    add_body = {"uris": uris}
    add_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    spotify_api_post(add_url, spotify_token, add_body)

    # Construir el URL final:
    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    return [playlist_url]

@app.route('/api/emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "")
        emotional_embedding, emotions = detect_user_emotions(message, n=3)
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
        # Devuelve un fallback “neutral” pero con 200 OK y CORS header
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
        genres        = data.get("genres", [])
        spotify_token = data.get("spotify_token")
        print(f"top genres: {genres}")
        songs_ids = get_playlist_ids(emotional_embedding_1, emotional_embedding_2, genres, k=5)
        urls = get_playlist_link(spotify_token, songs_ids)
        response      = generate_recommendation(ui1, ui2, br1, emo1, emo2, song="")
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

if __name__ == '__main__':
    app.run(debug=True)
