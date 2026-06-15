import os
import cv2
import torch
import numpy as np
import faiss
import sqlite3
from transformers import AutoImageProcessor, AutoModel
import json


model_name = "facebook/dinov2-base"

# Image embedding model and pre-processor
pre_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")
dimension = 768

# Create index and initialize database
index = faiss.IndexFlatL2(dimension)

db_conn = sqlite3.connect('mtg_cards.db')
cursor = db_conn.cursor()
cursor.execute("""
                CREATE TABLE IF NOT EXISTS cards 
                    (faiss_id INTEGER, 
                    scryfall_id TEXT,
                    name TEXT,
                    set_code TEXT
                )
            """)

def get_embedding(image_path):
    '''Pulls image from unique art reference folder to embed the cards art to a vector value

    Args:
        image_path: path to cards full art in reference_folder.

    Returns:
        embedding vector for faiss index

    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = pre_processor(images = image, return_tensors="pt").to("cuda")

    # Run embedding model for vector values
    with torch.no_grad():
        outputs = model(**inputs)

        return outputs.last_hidden_state[:,0,:].cpu().numpy()
    
# Scryfall API unique card json storage for full art images
json_file = "unique-artwork-20260314210622.json"

with open(json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

    card_lookup = {card['id']: card for card in data}

del data

ref_path = "reference_cards"

# Iterate throuh all images in reference_cards folder
for i, filename in enumerate(os.listdir(ref_path)):
    if not filename.lower().endswith(('.png', '.jpg', 'jpeg', 'webp')):
        continue
    
    # obtain card data from sryfall to fill index
    scryfall_id = os.path.splitext(filename)[0]

    card_data = card_lookup.get(scryfall_id)
    if card_data:
        name = card_data.get('name')
        set_code = card_data.get('set')
        
    full_path = os.path.join(ref_path, filename)
    print(f"Indexing {i}: {filename}")

    vector = get_embedding(full_path)

    index.add(vector.astype('float32'))

    cursor.execute("INSERT INTO cards (faiss_id, scryfall_id, name, set_code) VALUES (?, ?, ?, ?)", (i, scryfall_id, name, set_code))

faiss.write_index(index, "mtg_cards.index")
db_conn.commit()
db_conn.close()
print("Indexing and Database create sucesfully!")
