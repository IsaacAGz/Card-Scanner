from fastapi import FastAPI, UploadFile, File
import imutils
from transformers import AutoImageProcessor, AutoModel
from inference_sdk import InferenceHTTPClient
from contextlib import asynccontextmanager
from ultralytics import YOLO
from thefuzz import fuzz
import torch
import os
import faiss
import cv2
import numpy as np
import uvicorn
import sqlite3
import easyocr


model = None
processor = None
index = None
db_conn = None
reader = None
yolo_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, index, db_conn, reader, yolo_model

    print("Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    reader = easyocr.Reader(['en'], gpu = torch.cuda.is_available())

    if os.path.exists("best.pt"):
        print("Loading YOLO model...")
        yolo_model = YOLO("best.pt")
    else:
        raise FileNotFoundError("Yolo model file 'best.pt' not found.")

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

"""
Looks at the title of the card and checks if it matches the predicted name up to a threshold.
"""
def verify_with_ocr(warped_image, predicted_name):
    height, width = warped_image.shape[:2]

    title_crop = warped_image[0:int(height * 0.12), 0:width]

    ocr_result = reader.readtext(title_crop, detail = 0)
    detected_text = " ".join(ocr_result).strip().lower()
    predicted_name = predicted_name.lower()

    match_score = fuzz.partial_ratio(predicted_name, detected_text)

    return match_score >= 80


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

    results = yolo_model(frame, conf = 0.5, verbose = False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pad = 20
            x1 = max(0, x1-pad)
            y1 = max(0, y1-pad)
            x2 = min(frame.shape[1], x2+pad)
            y2 = min(frame.shape[0], y2+pad)

            roi = frame[y1:y2, x1:x2]


            grayscale_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)
            edged_image = cv2.Canny(blurred_image, 75, 200)

            contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

            #cv2.drawContours(frame, contours, -1, (0,255,0), 2)

            if not contours:
                continue

            largest_contour = max(contours, key = cv2.contourArea)

            perimeter = cv2.arcLength(largest_contour, True)
            approximation = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

            if len(approximation) == 4:
                points = approximation.reshape(4,2)

                points[:, 0] += x1
                points[:, 1] += y1

                rectangle = order_points(points)
                destination_points = np.array([[0,0], [300,0], [300,420], [0,420]], dtype = "float32")
                perspective_transform = cv2.getPerspectiveTransform(rectangle, destination_points)
                warped = cv2.warpPerspective(frame, perspective_transform, (300, 420))

                embedding = get_embedding(warped)
                embedding = np.array(embedding, dtype = np.float32)

                if len(embedding.shape) == 1:
                    embedding = np.expand_dims(embedding, axis = 0)
                
                distances, indices = index.search(embedding, 1)

                best_distance = distances[0][0]
                best_faiss_id= indices[0][0]

                card_info = get_card_info(best_faiss_id)

                if card_info:
                    predicted_name = card_info[0]

                    is_verified = verify_with_ocr(warped, predicted_name)

                    if is_verified:
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
    
    #frame = imutils.resize(frame, width = 1080)

    results = process_image(frame)

    return {"status": "success", "data": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)