"""Load embedding and identification runtimes outside the FastAPI app."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Callable

import faiss
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from transformers import AutoImageProcessor

FAISS_INDEX_PATH = "mtg_cards.index"
DIST_THRESHOLD = 300


@dataclass
class EmbeddingRuntime:
    get_embedding: Callable[[list], np.ndarray]


@dataclass
class IdentificationRuntime:
    get_embedding: Callable[[list], np.ndarray]
    identify_crops: Callable[[list, list, float], list[dict]]


def load_embedding_runtime() -> EmbeddingRuntime:
    load_dotenv()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession("onnx_dinov2/model.onnx", providers=providers)
    processor = AutoImageProcessor.from_pretrained("onnx_dinov2")

    def get_embedding(crop_list) -> np.ndarray:
        inputs = processor(images=crop_list, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        ort_outputs = ort_session.run(None, {"pixel_values": pixel_values})
        return ort_outputs[0][:, 0, :].astype("float32")

    return EmbeddingRuntime(get_embedding=get_embedding)


def load_identification_runtime(
    *,
    dist_threshold: float = DIST_THRESHOLD,
    faiss_index_path: str = FAISS_INDEX_PATH,
    db_path: str = "mtg_cards.db",
) -> IdentificationRuntime:
    load_dotenv()

    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index file '{faiss_index_path}' not found.")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession("onnx_dinov2/model.onnx", providers=providers)
    processor = AutoImageProcessor.from_pretrained("onnx_dinov2")
    faiss_index = faiss.read_index(faiss_index_path)
    db_conn = sqlite3.connect(db_path, check_same_thread=False)

    def get_card_info(faiss_id: int):
        if faiss_id < 0:
            return None
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT name, set_code, scryfall_id
            FROM cards
            WHERE faiss_id = ?
            """,
            (int(faiss_id),),
        )
        return cursor.fetchone()

    def get_embedding(crop_list) -> np.ndarray:
        inputs = processor(images=crop_list, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        ort_outputs = ort_session.run(None, {"pixel_values": pixel_values})
        return ort_outputs[0][:, 0, :].astype("float32")

    def identify_crops(crop_list, boxes, threshold: float = dist_threshold) -> list[dict]:
        if not crop_list:
            return []

        all_vectors = get_embedding(crop_list)
        all_distances, all_indices = faiss_index.search(all_vectors, k=1)

        detections: list[dict] = []
        for i, box in enumerate(boxes):
            dist_val = float(all_distances[i][0])
            card_idx = all_indices[i][0]
            card_info = get_card_info(card_idx)
            is_identified = dist_val <= threshold

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

    return IdentificationRuntime(
        get_embedding=get_embedding,
        identify_crops=identify_crops,
    )
