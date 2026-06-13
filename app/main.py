from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModel
from contextlib import asynccontextmanager
from ultralytics import YOLO
import torch
import os
import faiss
import cv2
import numpy as np
import uvicorn
import sqlite3

yolo = None
model = None
processor = None
index = None
db_conn = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, index, db_conn

    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO("yolo11s.pt")

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
    results = yolo(frame, save=True, conf=.75)

    found_cards_info = []

    height, width, _ = frame.shape

    for prediction in results[0].boxes:

        #Crop predicted card to get faiss index from embedding model

        xyxy = prediction.xyxy[0].tolist()
        
        xmin = max(0, int(xyxy[0]))
        ymin = max(0, int(xyxy[1]))
        xmax = min(width, int(xyxy[2]))
        ymax = min(height, int(xyxy[3]))

        card_crop = frame[ymin:ymax, xmin:xmax]
        
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
            "box": [xmin, ymin, xmax, ymax]
        })
    return {"count": len(found_cards_info), "cards": found_cards_info}


@app.post("/scan")
async def scan_cards(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Show window for testing
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = process_image(frame)

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)