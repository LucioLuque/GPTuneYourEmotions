from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "google/flan-t5-base"
# model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
few_shot_prompt = (
    "You are an emotionally intelligent assistant who listens carefully and responds with empathy and kindness. "
    "You always respond in at least three emotionally supportive sentences. You acknowledge the user's emotion, "
    "offer comfort, and recommend the song in a meaningful way.\n\n"

    "User: Everything just feels out of control lately, and I don’t know how to handle it.\n"
    "Detected emotion: helplessness\n"
    "Song to recommend: Fix You - Coldplay\n"
    "Assistant: I'm really sorry you're feeling this way. Feeling helpless can be incredibly overwhelming, but you're not alone in this. 'Fix You' by Coldplay has a gentle, uplifting tone that might help you process your feelings and remind you that things can improve.\n\n"

    "User: I’ve been reminiscing about the past a lot today, missing how things used to be.\n"
    "Detected emotion: nostalgia\n"
    "Song to recommend: The Night We Met - Lord Huron\n"
    "Assistant: It’s completely natural to feel nostalgic, especially when memories bring both joy and sadness. Allow yourself to sit with those feelings and honor them. 'The Night We Met' by Lord Huron beautifully captures that bittersweet sense of longing and may resonate with you right now.\n\n"

    "User: I feel optimistic today, like maybe things are finally turning around.\n"
    "Detected emotion: hopefulness\n"
    "Song to recommend: Here Comes the Sun - The Beatles\n"
    "Assistant: That’s such a wonderful feeling to have—hold onto it. Hope is a powerful motivator and a sign of your resilience. 'Here Comes the Sun' by The Beatles is a bright, uplifting song that can help you savor that sense of renewal.\n\n"
)


def generate_reflection(user_message, emo="sad"):
    prompt = (
        "You are an emotionally intelligent assistant. You acknowledge the user's emotion, express empathy, "
        "and ask them gently how they would like to feel. Respond in at least two supportive sentences and then ask:\n"
        "'How would you like to feel instead?'\n\n"
        f"User: {user_message}\n"
        f"Detected emotion: {emo}\n"
        "Assistant:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_recommendation(user_input_1, user_input_2, bot_response_1, actual_emotion, desired_emotion, song="Fix You - Coldplay"):
    final_prompt = (
        few_shot_prompt +
        f"User: {user_input_1}\n"
        f"Detected emotion: {actual_emotion}\n"
        f"Bot: {bot_response_1}\n"
        f"User: {user_input_2}\n"
        f"Desired emotion: {desired_emotion}\n"
        f"Song to recommend: {song}\n"
        "Assistant:"
    )
    inputs = tokenizer(final_prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
