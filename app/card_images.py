"""Build ZIP archives of Scryfall card images from detected card metadata."""

from __future__ import annotations

import io
import re
import sqlite3
import time
import zipfile

import requests

SCRYFALL_HEADERS = {
    "User-Agent": "GRXSCardScannerMicro-service",
    "Accept": "application/json",
}


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", value).strip().replace(" ", "_")
    return cleaned or "card"


def lookup_scryfall_id(conn: sqlite3.Connection, name: str, set_code: str) -> str | None:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT scryfall_id
        FROM cards
        WHERE name = ? AND set_code = ?
        LIMIT 1
        """,
        (name, set_code),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def fetch_card_image_url(name: str, set_code: str, scryfall_id: str | None = None) -> str:
    if scryfall_id:
        response = requests.get(
            f"https://api.scryfall.com/cards/{scryfall_id}",
            headers=SCRYFALL_HEADERS,
            timeout=20,
        )
    else:
        response = requests.get(
            "https://api.scryfall.com/cards/named",
            params={"fuzzy": name, "set": set_code},
            headers=SCRYFALL_HEADERS,
            timeout=20,
        )

    if response.status_code == 429:
        time.sleep(2)
        return fetch_card_image_url(name, set_code, scryfall_id)

    response.raise_for_status()
    card = response.json()

    image_uris = card.get("image_uris")
    if not image_uris and card.get("card_faces"):
        image_uris = card["card_faces"][0].get("image_uris")

    if not image_uris or "border_crop" not in image_uris:
        raise ValueError(f"No border_crop image available for {name} ({set_code}).")

    return image_uris["border_crop"]


def download_image(url: str) -> bytes:
    response = requests.get(url, headers=SCRYFALL_HEADERS, timeout=20)
    if response.status_code == 429:
        time.sleep(2)
        return download_image(url)
    response.raise_for_status()
    return response.content


def build_card_images_zip(cards: list[dict], conn: sqlite3.Connection) -> bytes:
    unique_cards: dict[tuple[str, str], dict] = {}
    for card in cards:
        name = card.get("name")
        set_code = card.get("set")
        if not name or not set_code:
            continue
        unique_cards[(name, set_code)] = card

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for (name, set_code), _card in unique_cards.items():
            scryfall_id = lookup_scryfall_id(conn, name, set_code)
            image_url = fetch_card_image_url(name, set_code, scryfall_id)
            image_bytes = download_image(image_url)

            filename = f"{set_code}/{sanitize_filename(name)}.jpg"
            archive.writestr(filename, image_bytes)
            time.sleep(0.1)

    buffer.seek(0)
    return buffer.getvalue()
