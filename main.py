from fastapi import FastAPI, UploadFile, File
import imutils
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

def order_points(points):
    rectangle = np.zeros((4,2), dtype = "float32")
    sum = points.sum(axis = 1)
    rectangle[0] = points[np.argmin(sum)]
    rectangle[2] = points[np.argmax(sum)]

    diff = np.diff(points, axis = 1)
    
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]

    return rectangle

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
    found_cards_info = []

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)
    edged_image = cv2.Canny(blurred_image, 75, 200)

    contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

    #cv2.drawContours(frame, contours, -1, (0,255,0), 2)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approximation) == 4:
            points = approximation.reshape(4,2)
            rectangle = order_points(points)
            destination_points = np.array([[0,0], [420,0], [420,300], [0,300]], dtype = "float32")
            perspective_transform = cv2.getPerspectiveTransform(rectangle, destination_points)
            warped = cv2.warpPerspective(frame, perspective_transform, (420, 300))

            embedding = get_embedding(warped)
            embedding = np.array(embedding, dtype = np.float32)

            if len(embedding.shape) == 1:
                embedding = np.expand_dims(embedding, axis = 0)
            
            distances, indices = index.search(embedding, 1)

            faiss_id = indices[0][0]

            card_info = get_card_info(faiss_id)

            if card_info:
                found_cards_info.append({
                    "name": card_info[0],
                    "set_code": card_info[1],
                    "distance": float(distances[0][0]),
                    "box": rectangle.astype(int).tolist()
                })
            
    return found_cards_info

@app.post("/scan")
async def scan_cards(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = imutils.resize(frame, width = 600)

    results = process_image(frame)

    return {"status": "success", "data": results}

@app.post("/scan_multiple")
async def scan_multiple(file: UploadFile = File(...)):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)