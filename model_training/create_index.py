import os
import cv2
import torch
import numpy as np
import faiss
import sqlite3
import urllib.request
from transformers import AutoImageProcessor
import json
import onnxruntime as ort
import time

# Image embedding model and pre-processor
print("Loading optimized ONNX DINOv2 model session...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession("../onnx_dinov2/model.onnx", providers=providers)
pre_processor = AutoImageProcessor.from_pretrained("../onnx_dinov2")

dimension = 768

# Create index and initialize database
index = faiss.IndexFlatL2(dimension)

db_conn = sqlite3.connect('mtg_cards.db')
cursor = db_conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS cards (
        faiss_id INTEGER PRIMARY KEY, 
        scryfall_id TEXT,
        name TEXT,
        set_code TEXT
    )
""")

def get_embedding_onnx(image_np):
    '''Generates an vector embedding matrix using local ONNX architecture runtime.

    Args:
        image_np:

    Returns:
        embedding vector for faiss index
    '''
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    inputs = pre_processor(images=rgb_image, return_tensors="np")

    # Format tensor payload directly for ONNX inference graph
    pixel_values = inputs["pixel_values"].astype(np.float32)
    ort_inputs = {"pixel_values": pixel_values}
    ort_outputs = ort_session.run(None, ort_inputs)

    embedding = ort_outputs[0][:, 0, :].astype('float32')

    return embedding

json_file = "unique-artwork-20260622090332.json"

print(f"Reading Scryfall artwork dataset metadata from {json_file}...")
with open(json_file, 'r', encoding='utf-8') as file:
    card_data_list = json.load(file)

print(f"Total entries lodade: {len(card_data_list)}. Beginning Vector Compilation...")

faiss_id_counter = 0

# Iterate throuh all cards in reference_cards folder
for i, card in enumerate(card_data_list):
    if card.get('digital') is True:
        continue

    image_uris = card.get('image_uris')

    if not image_uris or 'border_crop' not in image_uris:

        card_faces = card.get('card_faces')
        if card_faces and 'image_uris' in card_faces[0]:
            image_uris = card_faces[0]['image_uris']
        else:
            continue

    scryfall_id = card.get('id')
    name = card.get('name')
    set_code = card.get('set')
    image_url = image_uris['border_crop']

    if i % 100 == 0:
        print(f"Processing Progress: Checked {i}/{len(card_data_list)} | Indexed Rows: {faiss_id_counter}")
    
    try:
        req = urllib.request.Request(image_url, headers={'User-Agent': 'MTG-Scanner-Builder/1.0'})

        with urllib.request.urlopen(req) as response:
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            card_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if card_img is None or card_img.size == 0:
            continue
        
        vector = get_embedding_onnx(card_img)

        index.add(vector)

        cursor.execute("""
            INSERT INTO cards (faiss_id, scryfall_id, name, set_code)
            VALUES (?, ?, ?, ?)
            """, (faiss_id_counter, scryfall_id, name, set_code))

        faiss_id_counter += 1

        if faiss_id_counter % 500 == 0:
            db_conn.commit()

            faiss.write_index(index, "mtg_cards.index")
        
        time.sleep(0.05)

    except Exception as e:
        print(f"\nSkipping card {name} due to unexpected download network errors: {e}")
        continue

print("\nWriting indexing structures to local files..")
faiss.write_index(index, "mtg_cards.index")
db_conn.commit()
db_conn.close()

print(f"Succesfully generated database structures! Final indexed count: {faiss_id_counter} cards.")
            

