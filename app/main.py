from fastapi import FastAPI, UploadFile, File, status, HTTPException, Header, Security
from transformers import AutoImageProcessor
from contextlib import asynccontextmanager
from ultralytics import YOLO
from dotenv import load_dotenv
import onnxruntime as ort
import requests
import time
import urllib.request
import torch
import os
import faiss
import cv2
import numpy as np
import uvicorn
import sqlite3

ort_session = None
processor = None
yolo = None
index = None
db_conn = None

load_dotenv()
API_KEY = os.getenv("ADMIN_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session, processor, index, db_conn, yolo

    # Load YOLO model for card object detection, using default model for now
    print("Loading YOLO detection weights...")
    yolo = YOLO("yolo11s.pt")

    print("Loading optimized ONNX models...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession("onnx_dinov2/model.onnx", providers=providers)
    processor = AutoImageProcessor.from_pretrained("onnx_dinov2")

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

def get_card_info(faiss_id: int):
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
    
def get_embedding(crop_list) -> np.ndarray:
    '''Generates a batch of DINOv2 embeddings using highly optimized ONNX Runtime execution.

    '''

    inputs = processor(images=crop_list, return_tensors="np")

    pixel_values = inputs["pixel_values"].astype(np.float32)

    ort_inputs = {"pixel_values": pixel_values}
    ort_outputs = ort_session.run(None, ort_inputs)
        
    embeddings = ort_outputs[0][:, 0, :].astype('float32')

    return embeddings


def fetch_new_cards(set_code: str):
    '''
    Helper funciton that performs get request to srycall API to fetch all cards from selected 
    set and returns as a list of dictionaries (cards) to be downloaded and embedded.

    Args:
        set_code: string that matches a MTG set, usually newly released sets.

    Returns:
        cards_to_index: list of dictionaries containing the card data for all cards in the set.
    '''
    url = f"https://api.scryfall.com/cards/search?q=set:{set_code.lower()}+is:unique"
    headers = {"User-Agent": "GRXSCardScannerMicro-service", "Accept": "application/json"}

    cards_to_index = []

    while url:
        response = requests.get(url, headers=headers)

        # Too many requests
        if response.status_code == 429:
            time.sleep(2)
            continue

        if response.status_code != 200:
            break
        
        data = response.json()

        # Iterate through all cards to see if eligible to embedd and retrieve all data and uri for card image.
        for card in data.get('data', []):
            if 'image_uris' in card and 'border_crop' in card['image_uris'] and 'paper' in card['games']:
                cards_to_index.append({
                    "id": card['id'],
                    "name": card['name'],
                    "set_code": card['set'],
                    "image_url": card['image_uris']['border_crop']
                })
        
        url = data.get('next_page') if data.get('has_more') else None
        time.sleep(0.1)

    return cards_to_index


def sync_new_cards(set_code: str, faiss_index, sqlite_conn):
    '''

    '''
    new_cards = fetch_new_cards(set_code)
    print(f"Found {len(new_cards)} new cards to index.")

    cursor = sqlite_conn.cursor()

    valid_cards_metadata = []
    crop_list = []

    for card in new_cards:
        cursor.execute("SELECT 1 FROM cards WHERE name = ? AND set_code = ?", (card['name'], card['set_code']))

        if cursor.fetchone():
            continue

        try:
            resp = urllib.request.urlopen(card['image_url'])
            img_array = np.array(bytearray(resp.read()), dtype=np.uint8)
            card_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if card_img is None:
                continue

            rgb_image = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
            crop_list.append(rgb_image)

            valid_cards_metadata.append(card)
            
        except Exception as e:
            print(f"Failed to process card{card['name']}: {e}")
            continue
    
    if not crop_list:
        print("No new unique cards to index.")
        return

    print(f"Generating embeddings for a batch of {len(crop_list)} cards...")
    
    all_vectors = get_embedding(crop_list)

    start_faiss_id = faiss_index.ntotal
    faiss_index.add(all_vectors)

    for i, card in enumerate(crop_list):
        assigned_faiss_id = start_faiss_id + i

        cursor.execute("""
            INSERT INTO cards (faiss_id, scryfall_id, name, set_code)
            VALUES (?, ?, ?)
            """, (assigned_faiss_id, card['id'], card['name'], card['set_code']))
    
    sqlite_conn.commit()
    print(f"Succesfully batch-indexed {len(valid_cards_metadata)} cards into database and FAISS.")

                

def process_image(frame):
    '''Crops image, performs vector embedding, to retrieve card attributes.

    Args: 
        image as numpy array 

    Returns:
        Dictionary: containing number of cards and and the information of the cards obtained
    '''
    results = yolo(frame, save=True, conf=.75)

    found_cards_info = []
    found_boxes = []
    crop_list = []

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

        rgb_crop = cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB)
        crop_list.append(rgb_crop)
        found_boxes.append([xmin, ymin, xmax, ymax])

    if not crop_list:
        return {"count": 0, "cards": []}

    # FAISS ID embedding vector
    all_vectors = get_embedding(crop_list) 

    all_distances, all_indices = index.search(all_vectors, k=1)

    # Lookup cards by searching vector space for all cards(vectors found)
    for i, box in enumerate(found_boxes):
        dist_val = float(all_distances[i][0])

        # Card distance threshold
        if dist_val > 300:
            continue

        card_idx = all_indices[i][0]

        card_info = get_card_info(card_idx)

        # Append found card and card attributes if database card close enough to cropped embedded card
        found_cards_info.append({
            "name": card_info[0] if card_info else "Unknown",
            "set": card_info[1] if card_info else "Unknown",
            "dist": dist_val,
            "box": box
        })

    return {"count": len(found_cards_info), "cards": found_cards_info}




@app.get("/health")
async def health():
    '''Health check for API'''
    return {"status": "healthy"}

@app.post("/admin/sync-set")
async def sync_set(set_code: str, x_api_key: str = Header(...)):
    '''
    Admin endpoint to sync set cards to index db

    Args:
        set_code: string specifying desired set to sync
        x_api_key: string to verify admin, stored in header
    '''
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized execution attempt.")
    
    sync_new_cards(set_code, index, db_conn)

    get_card_info.cache_clear()

    return {"message": f"Set {set_code.upper()} successfully, synced into lookup space."}


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
            status_code=status.HTTP_400_BAD_REQUEST,
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