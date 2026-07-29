from fastapi import FastAPI, UploadFile, File, status, HTTPException, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoImageProcessor
from contextlib import asynccontextmanager
from ultralytics import YOLO
from dotenv import load_dotenv
import onnxruntime as ort
import requests
import time
import urllib.request
import os
import faiss
import cv2
import numpy as np
import uvicorn
import sqlite3
import tempfile

from card_images import build_card_images_zip
from video_scan import detect_card_boxes, process_video

ort_session = None
processor = None
yolo = None
index = None
db_conn = None

load_dotenv()
API_KEY = os.getenv("ADMIN_KEY")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "mtg_yolo_best.pt")
MAX_VIDEO_BYTES = 100 * 1024 * 1024
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DIST_THRESHOLD = 300
FAISS_INDEX_PATH = "mtg_cards.index"
DEFAULT_YOLO_CONF = 0.75

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session, processor, index, db_conn, yolo

    print(f"Loading YOLO detection weights from {YOLO_WEIGHTS}...")
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"YOLO weights file '{YOLO_WEIGHTS}' not found.")
    yolo = YOLO(YOLO_WEIGHTS)

    print("Loading optimized ONNX models...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession("onnx_dinov2/model.onnx", providers=providers)
    processor = AutoImageProcessor.from_pretrained("onnx_dinov2")

    # FAISS Index initializer
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        raise FileNotFoundError(f"FAISS index file '{FAISS_INDEX_PATH}' not found.")


    # Card DB 
    db_conn = sqlite3.connect('mtg_cards.db', check_same_thread=False)
    print("Assets loaded successfully.")

    yield
    db_conn.close()

app = FastAPI(lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

class CardRef(BaseModel):
    name: str
    set: str

class CardImagesRequest(BaseModel):
    cards: list[CardRef] = Field(min_length=1)

def get_card_info(faiss_id: int):
    ''' Uses faiss id in database to pull card attributes.

    Args: 
        int: value for db index lookup

    Returns: 
        tuple: name, set_code, scryfall_id

    '''
    if faiss_id < 0:
        return None
    cursor  = db_conn.cursor()
    cursor.execute("""
                    SELECT name, set_code, scryfall_id
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


def identify_crops(crop_list, boxes, dist_threshold: float = DIST_THRESHOLD) -> list[dict]:
    '''Embed crops and identify cards via FAISS + SQLite lookup.

    Returns one entry per detected box, including unmatched cards.
    '''
    if not crop_list:
        return []

    all_vectors = get_embedding(crop_list)
    all_distances, all_indices = index.search(all_vectors, k=1)

    detections: list[dict] = []
    for i, box in enumerate(boxes):
        dist_val = float(all_distances[i][0])
        card_idx = all_indices[i][0]
        card_info = get_card_info(card_idx)
        is_identified = dist_val <= dist_threshold

        detection = {
            "box": box,
            "identified": is_identified,
            "dist": dist_val,
            "name": None,
            "set": None,
            "scryfall_id": None,
        }

        if is_identified:
            detection["name"] = card_info[0] if card_info else "Unknown"
            detection["set"] = card_info[1] if card_info else "Unknown"
            detection["scryfall_id"] = card_info[2] if card_info else None

        detections.append(detection)

    return detections


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


def sync_new_cards(set_code: str, faiss_index, sqlite_conn) -> int:
    '''Fetch, embed, and index cards from a Scryfall set not already in the database.'''
    new_cards = fetch_new_cards(set_code)
    print(f"Found {len(new_cards)} cards from Scryfall for set {set_code}.")

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
            print(f"Failed to process card {card['name']}: {e}")
            continue
    
    if not crop_list:
        print("No new unique cards to index.")
        return 0

    print(f"Generating embeddings for a batch of {len(crop_list)} cards...")
    
    all_vectors = get_embedding(crop_list)

    start_faiss_id = faiss_index.ntotal
    faiss_index.add(all_vectors)

    for i, card in enumerate(valid_cards_metadata):
        assigned_faiss_id = start_faiss_id + i

        cursor.execute("""
            INSERT INTO cards (faiss_id, scryfall_id, name, set_code)
            VALUES (?, ?, ?, ?)
            """, (assigned_faiss_id, card['id'], card['name'], card['set_code']))
    
    sqlite_conn.commit()
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print(f"Successfully batch-indexed {len(valid_cards_metadata)} cards into database and FAISS.")
    return len(valid_cards_metadata)

def process_image(
    frame,
    save_yolo: bool = False,
    conf: float = DEFAULT_YOLO_CONF,
    dist_threshold: float = DIST_THRESHOLD,
):
    '''Crops image, performs vector embedding, to retrieve card attributes.

    Args: 
        image as numpy array 

    Returns:
        Dictionary: containing number of cards and and the information of the cards obtained
    '''
    boxes, crop_list = detect_card_boxes(frame, yolo, conf=conf, save_yolo=save_yolo)

    if not crop_list:
        return {
            "detected_count": 0,
            "identified_count": 0,
            "count": 0,
            "cards": [],
            "detections": [],
        }

    detections = identify_crops(crop_list, boxes, dist_threshold=dist_threshold)
    found_cards_info = [
        {
            "name": detection["name"],
            "set": detection["set"],
            "dist": detection["dist"],
            "box": detection["box"],
        }
        for detection in detections
        if detection["identified"]
    ]

    return {
        "detected_count": len(detections),
        "identified_count": len(found_cards_info),
        "count": len(found_cards_info),
        "cards": found_cards_info,
        "detections": detections,
    }


def run_process_video(
    path: str,
    frame_stride: int,
    max_frames: int,
    conf: float = DEFAULT_YOLO_CONF,
    dist_threshold: float = DIST_THRESHOLD,
) -> dict:
    return process_video(
        path,
        yolo=yolo,
        identify_crops=identify_crops,
        frame_stride=frame_stride,
        max_frames=max_frames,
        conf=conf,
        dist_threshold=dist_threshold,
    )


@app.get("/health")
async def health():
    '''Health check for API'''
    return {"status": "healthy"}


@app.get("/ui")
async def ui_redirect():
    return RedirectResponse(url="/ui/")


if os.path.isdir(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

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
    
    added_count = sync_new_cards(set_code, index, db_conn)

    return {
        "message": f"Set {set_code.upper()} synced into lookup space.",
        "cards_added": added_count,
    }


@app.post("/scan", status_code=status.HTTP_200_OK)
async def scan_cards(
    file: UploadFile = File(...),
    conf: float = DEFAULT_YOLO_CONF,
    dist_threshold: float = DIST_THRESHOLD,
):
    '''Scanning card endpoint for API

    Args:
        Uploaded Image

    Returns:
        json with number of cards and information of found cards

    '''
    if conf <= 0 or conf > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="conf must be between 0 and 1.")
    if dist_threshold <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="dist_threshold must be > 0.")

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

    results = process_image(frame, conf=conf, dist_threshold=dist_threshold)

    return results


@app.post("/scan/video", status_code=status.HTTP_200_OK)
async def scan_video(
    file: UploadFile = File(...),
    frame_stride: int = 5,
    max_frames: int = 300,
    conf: float = DEFAULT_YOLO_CONF,
    dist_threshold: float = DIST_THRESHOLD,
):
    '''Scan a video and return unique cards identified across sampled frames.'''
    if not file.filename or not file.filename.lower().endswith(VIDEO_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Upload MP4, MOV, AVI, MKV, or WEBM.",
        )

    if frame_stride < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="frame_stride must be >= 1.")
    if max_frames < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="max_frames must be >= 1.")
    if conf <= 0 or conf > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="conf must be between 0 and 1.")
    if dist_threshold <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="dist_threshold must be > 0.")

    contents = await file.read()
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video exceeds max size of {MAX_VIDEO_BYTES // (1024 * 1024)} MB.",
        )

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        results = await run_in_threadpool(
            run_process_video,
            tmp_path,
            frame_stride,
            max_frames,
            conf,
            dist_threshold,
        )
        return results
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        os.unlink(tmp_path)


@app.post("/cards/images-zip")
async def download_card_images(payload: CardImagesRequest):
    '''Return a ZIP of Scryfall border_crop images for the requested cards.'''
    try:
        zip_bytes = await run_in_threadpool(
            build_card_images_zip,
            [card.model_dump() for card in payload.cards],
            db_conn,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch card images: {exc}",
        ) from exc

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=detected_cards.zip"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
