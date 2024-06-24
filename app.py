from PIL import Image
import requests
import numpy as np
import pickle
import torch

import cv2

from preprocessing import device, processor, model
from video_processing import extract_frames

def load_image_PIL(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)

    # Compute the L2 norm
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def predict(image, average_positive_vector, average_negative_vector, is_url=True):
    if is_url:
        image1 = load_image_PIL(image)
    else:
        image1 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert frame to PIL image

    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        image_features1 = model.get_image_features(**inputs1)
    image_vector = image_features1.numpy()
    positive_similarity = cosine_similarity(average_positive_vector, np.transpose(image_vector))
    negative_similarity = cosine_similarity(average_negative_vector, np.transpose(image_vector))
    aesthetic_score = (+1*positive_similarity) + (-1*negative_similarity)
    return aesthetic_score*1000 #For Scale & Comparibility

def main():
    with open("animefood_positive_prompt.pkl", "rb") as f:
        average_positive_vector = pickle.load(f)
    with open("animefood_negative_prompt.pkl", "rb") as f:
        average_negative_vector = pickle.load(f)

    url = "https://img.freepik.com/free-photo/anime-style-clouds_23-2151071778.jpg?t=st=1719202751~exp=1719206351~hmac=1c5ee047b9ad13edc68f938c7a59693d7df833fbbffab79ce4ad77502ab4893b&w=1060"
    print(predict(url, average_positive_vector, average_negative_vector))

    url = "https://www.watchmojo.com/uploads/blipthumbs/WM-Anime-Top10-Ugliest-Anime-Characters_Y0T6C9-ALT_480.jpg"
    print(predict(url, average_positive_vector, average_negative_vector))

    url = "https://art.ngfiles.com/images/2282000/2282161_xenormxdraws_boa-hancock-one-piece.png?f1641570944"
    print(predict(url, average_positive_vector, average_negative_vector))

    url = "https://static1.cbrimages.com/wordpress/wp-content/uploads/2021/08/One-Piece-Wano-.jpg"
    print(predict(url, average_positive_vector, average_negative_vector))

    frames = extract_frames('test.mp4')
    aesthetic_counter = {}
    frames_list = []
    
    for idx, frame in enumerate(frames):
        result = predict(frame, average_positive_vector, average_negative_vector, is_url=False)
        aesthetic_counter[idx] = result
        frames_list.append(frame)
    
    max_val = float('-inf')
    result_indices = []
    
    for key, value in aesthetic_counter.items():
        if value > max_val:
            max_val = value
            result_indices = [key]  # Start a new list with the current key
        elif value == max_val:
            result_indices.append(key)  # Add to the list of best frames

    for idx in result_indices:
        winner = frames_list[idx]
        cv2.imshow("Most Aesthetic Frames", winner)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(len(result_indices))
    print(max_val)

if __name__ == "__main__":
    main()
