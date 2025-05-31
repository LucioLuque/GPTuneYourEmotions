# from model import generate_reflection, generate_recommendation
from emotions import detect_user_emotions, get_k_mid_points, emotions_labels, filtered_labels, format_top_emotions, get_playlist_ids
# from song import get_song

msg_1 = "I feel so lost and helpless right now, like nothing is going right."
msg_2 = "I wish I could just feel happy again, like I used to."
emotional_embedding_1, top_emotions_detected_1 = detect_user_emotions(msg_1, n=3)
emotional_embedding_2, top_emotions_detected_2 = detect_user_emotions(msg_2, n=3)

songs_ids = get_playlist_ids(emotional_embedding_1, emotional_embedding_2, None, k=5)

print(songs_ids)


# mid_points = get_k_mid_points(emotional_embedding_1, emotional_embedding_2, k=5)
# for i, mid_point in enumerate(mid_points):
#     dict_emotions = {label: mid_point[idx] for idx, label in enumerate(emotions_labels)}
#     if i==0:
#         print(f"Actual emotions: {dict_emotions}")
         
       
#     if i==len(mid_points)-1:
#         print(f"Desired emotions: {dict_emotions}")
#     else:
#         print(f"Mid point {i+1}: {dict_emotions}")
#     top_emotions = format_top_emotions(mid_point, filtered_labels, top_n=3)
#     print(f"Top emotions at mid point {i+1}: {top_emotions}")

