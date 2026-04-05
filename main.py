from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModel
from inference_sdk import InferenceHTTPClient
from contextlib import asynccontextmanager
import torch
import os
import faiss
import cv2
import numpy as np
import uvicorn
import sqlite3



CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="XCnO8a9HYvPhkX9nmxge"
)

model = None
processor = None
index = None
db_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, index, db_conn

    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    if os.path.exists("mtg_cards.index"):
        print("Loading FAISS index...")
        index = faiss.read_index("mtg_cards.index")
    else:
        raise FileNotFoundError("FAISS index file 'mtg_cards.index' not found.")

    db_conn = sqlite3.connect('mtg_cards.db', check_same_thread=False)
    print("Assets loaded successfully.")

    yield
    db_conn.close()

app = FastAPI(lifespan=lifespan)

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
    
def get_embedding(image_np):
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].cpu().numpy()

def process_image(frame):
    result = CLIENT.infer(frame, model_id="mtg-card-detection-slnj6/2")

    found_cards_info = []

    for prediction in result.get("predictions", []):
        x = prediction['x']
        y = prediction['y']
        w = prediction['width']
        h = prediction['height']

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        card_crop = frame[max(0, y1):y2, max(0, x1):x2]
        
        if card_crop.size == 0: continue

        vector = get_embedding(card_crop) 

        query_vector = vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vector, k=1)

        card_idx = indices[0][0]
        card_info = get_card_info(card_idx)
        found_cards_info.append({
            "name": card_info[0] if card_info else "Unknown",
            "set": card_info[1] if card_info else "Unknown",
            "dist": float(distances[0][0]),
            "box": [x1, y1, x2, y2]
        })
    return {"count": len(found_cards_info), "cards": found_cards_info}


@app.post("/scan")
async def scan_cards(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = process_image(frame)

    return results

@app.post("/scan_multiple")
async def scan_multiple(file: UploadFile = File(...)):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)