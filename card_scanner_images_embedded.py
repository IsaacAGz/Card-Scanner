import cv2
import numpy as np
import faiss
import os
import sqlite3
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="XCnO8a9HYvPhkX9nmxge"
)

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

index = faiss.read_index("mtg_cards.index")
db_conn = sqlite3.connect("mtg_cards.db")


def scan_multiple_cards(image_path, original_image):
    result = CLIENT.infer(image_path, model_id="mtg-card-detection-slnj6/2")
    
    found_cards_info = []

    for prediction in result["predictions"]:
        x = prediction['x']
        y = prediction['y']
        w = prediction['width']
        h = prediction['height']

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        card_crop = original_image[max(0, y1):y2, max(0, x1):x2]
        
        if card_crop.size == 0: continue

        vector = get_embedded(card_crop)
        query_vector = vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vector, k=1)

        card_idx = indices[0][0]
        dist = distances[0][0]
        
        card_info = get_card_info(card_idx)
        found_cards_info.append({
            "info": card_info, 
            "dist": dist, 
            "box": (x1, y1, x2, y2)
        })

    return found_cards_info

def get_embedded(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images = rgb_image, return_tensors = "pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].cpu().numpy()

def get_card_info(faiss_id):
    if faiss_id < 0:
        return None
    cursor  = db_conn.cursor()
    cursor.execute("""
                    SELECT name, set_code
                   FROM cards
                   where faiss_id = ?                   
                """,
                (int(faiss_id),))
    return cursor.fetchone()

input_folder = "images_to_scan"
os.makedirs(input_folder, exist_ok = True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".png",".jpg",".jpeg", ".webp")):
        continue

    image_path = os.path.join(input_folder, filename)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Coult not red {filename}")
        continue

    cards_found = scan_multiple_cards(image_path, frame)

    if not cards_found:
        print(f"No cards found in {filename}")
        continue

    
    for card in cards_found:
        if card["info"]:
            name, set_code = card["info"]
            display_text = f"{name} ({set_code})"
            print(f"File: {filename} -> MATCH: {display_text} | Distance: {card['dist']:.2f}")
        else:
            display_text = "Unknown Card"
            print(f"FILE: {filename} -> No match found in DB.")

        x1, y1, x2, y2 = card["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(input_folder, f"analyzed_{filename}")
    cv2.imwrite(output_path, frame)

db_conn.close()
print("Batch processing complete!")