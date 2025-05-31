from flask import Flask, request, jsonify
from flask_cors import CORS
from backend_factory import generate_reflection, generate_recommendation
from emotions import detect_user_emotions
from song import get_song

app = Flask(__name__)
CORS(app)

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
        print(f"top genres: {genres}")
        print(f"emotional_embedding_1: {emotional_embedding_1}")
        print(f"emotional_embedding_2: {emotional_embedding_2}")
        song          = get_song(emotional_embedding_1, emotional_embedding_2, genres)
        response      = generate_recommendation(ui1, ui2, br1, emo1, emo2, song)
        return jsonify({"response": response}), 200
    except Exception as e:
        app.logger.exception("Error en /api/recommend:")
        return jsonify({
            "response": "No pude generar la recomendación.",
            "error": str(e)
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
