from fastapi import FastAPI, UploadFile, File, status, HTTPException
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
    global yolo, model, processor, index, db_conn

    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YOLO model for card object detection
    yolo = YOLO("yolo11s.pt")

    # embedding model and image processor for vector embeddings
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    # FAISS Index initializer
    if os.path.exists("mtg_cards.index"):
        print("Loading FAISS index...")
        index = faiss.read_index("mtg_cards.index")
    else:
        raise FileNotFoundError("FAISS index file 'mtg_cards.index' not found.")


    # Card DB 
    db_conn = sqlite3.connect('mtg_cards.db', check_same_thread=False)
    print("Assets loaded successfully.")

    yield
    db_conn.close()

app = FastAPI(lifespan=lifespan)

def get_card_info(faiss_id):
    ''' Uses faiss id in database to pull card attributes.

    Args: 
        int: value for db index lookup

    Returns: 
        int: sql cursor query values: name, set_code

    '''
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
    '''

    '''

    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].cpu().numpy()

def process_image(frame):
    '''Crops image, performs vector embedding, to retrieve card attributes.

    Args: 
        image as numpy array 

    Returns:
        Dictionary: containing number of cards and and the information of the cards obtained
    '''

    results = yolo(frame, save=True, conf=.75)

    found_cards_info = []

    height, width, _ = frame.shape

    for prediction in results[0].boxes:

        # Crop predicted card to get faiss index from embedding model
        xyxy = prediction.xyxy[0].tolist()
        
        # Card/s boundaries in image
        xmin = max(0, int(xyxy[0]))
        ymin = max(0, int(xyxy[1]))
        xmax = min(width, int(xyxy[2]))
        ymax = min(height, int(xyxy[3]))

        card_crop = frame[ymin:ymax, xmin:xmax]
        
        if card_crop.size == 0: continue

        # FAISS ID embedding vector
        vector = get_embedding(card_crop) 

        # Lookup card by searching vector space
        query_vector = vector.reshape(1, -1).astype('float32')
        distances, indices = index.search(query_vector, k=1)

        # Card distance threshold
        if distances[0][0] > 800:
            continue

        card_idx = indices[0][0]
        card_info = get_card_info(card_idx)

        # Obtain card attributes if database card close enough to corpped card
        found_cards_info.append({
            "name": card_info[0] if card_info else "Unknown",
            "set": card_info[1] if card_info else "Unknown",
            "dist": float(distances[0][0]),
            "box": [xmin, ymin, xmax, ymax]
        })

    return {"count": len(found_cards_info), "cards": found_cards_info}


@app.get("/health")
async def health():
    '''Health check for API'''
    return {"status": "healthy"}


@app.post("/scan", status_code=status.HTTP_200_OK)
async def scan_cards(file: UploadFile = File(...)):
    '''Scanning card endpoint for API

    Args:
        Uploaded Image

    Returns:
        json with number of cards and information of found cards

    '''
    if not file.filename.lower().endswith(('.png','.jpg','.jpeg','.webp')):
        raise HTTPException(
            satus_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a PNG or JEPG image."
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode the uploading image file."
        )

    results = process_image(frame)

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)