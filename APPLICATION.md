# MTG Card Scanner — Application Overview

## What It Is

MTG Card Scanner is a FastAPI web service that **detects and identifies Magic: The Gathering cards from photos and videos**. You upload an image or video clip; the service finds card regions, compares them against a large library of known card artwork, and returns card names and sets.

Typical use cases:

- Scanning physical cards with a phone or webcam instead of typing names
- Building inventory from video of a collection or trade binder
- Downloading reference images of detected cards from Scryfall

---

## How It Works

The pipeline has two stages: **detection** (where are the cards?) and **identification** (which cards are they?).

```
Upload image/video
       │
       ▼
┌──────────────────┐
│ YOLO11 detection │  Custom fine-tuned model finds card bounding boxes
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Crop each card   │  Axis-aligned crop from bounding box
└────────┬─────────┘
         ▼
┌──────────────────┐
│ DINOv2 embedding │  ONNX Runtime; facebook/dinov2-base
└────────┬─────────┘
         ▼
┌──────────────────┐
│ FAISS search     │  Nearest-neighbor lookup in mtg_cards.index
└────────┬─────────┘
         ▼
┌──────────────────┐
│ SQLite lookup    │  faiss_id → name, set, scryfall_id (mtg_cards.db)
└────────┬─────────┘
         ▼
    JSON response
```

For **video**, sampled frames are processed with **track-based deduplication**: if the same card stays in frame, it is identified once per track (IoU matching), not on every frame. The response contains **unique cards only**.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn |
| Object detection | Ultralytics YOLO11 (custom `mtg_yolo_best.pt`) |
| Embeddings | Facebook DINOv2 via ONNX Runtime |
| Vector search | FAISS (`IndexFlatL2`) |
| Metadata | SQLite (`mtg_cards.db`) |
| Card data source | Scryfall API (index build + image zip) |
| Image/video I/O | OpenCV |
| Deployment | Docker, Makefile |
| Training (local) | `model_training/` — YOLO fine-tune, Label Studio import |

---

## Project Structure

```
Card Scanner/
├── app/
│   ├── main.py              # FastAPI app, endpoints, shared inference
│   ├── video_scan.py        # Video processing + track deduplication
│   ├── card_images.py       # Scryfall ZIP builder
│   ├── build_onnx.py        # One-time DINOv2 → ONNX export
│   ├── static/              # Web UI (index.html, app.js, style.css)
│   └── mtg_yolo_best.pt     # Fine-tuned YOLO weights (~18 MB)
├── app_testing/
│   ├── test_client.py       # Webcam → /scan loop
│   └── test_video_client.py # Video file → /scan/video
├── scripts/
│   ├── check_artifacts.py   # Validate required runtime files
│   ├── setup.ps1            # Windows one-time setup
│   └── setup.sh             # macOS/Linux one-time setup
├── model_training/          # Gitignored — dataset, train.py, import scripts
├── mtg_cards.db             # SQLite card metadata (~3 MB)
├── mtg_cards.index          # FAISS index (~152 MB, gitignored)
├── onnx_dinov2/             # ONNX model + processor (gitignored)
├── docker-compose.yml       # Docker with artifact volume mounts
├── .env.example             # Environment variable template
├── requirements.txt
├── Dockerfile
├── Makefile
├── README.md                # Setup and API usage
└── APPLICATION.md           # This file
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/ui` | Web UI for scanning and ZIP download |
| POST | `/scan` | Upload image; returns detected cards with boxes |
| POST | `/scan/video` | Upload video; returns unique cards across sampled frames |
| POST | `/cards/images-zip` | JSON list of cards → ZIP of Scryfall border_crop images |
| POST | `/admin/sync-set` | Add cards from a new MTG set to the index (API key required) |

Interactive docs: `http://localhost:8000/docs`

---

## Current State

### Implemented and working

- **Image scanning** (`POST /scan`) with bounding boxes, name, set, and distance
- **Video scanning** (`POST /scan/video`) with frame stride, max frames cap, and track deduplication
- **Card image ZIP** (`POST /cards/images-zip`) from detected card names/sets
- **Custom YOLO model** loaded via `YOLO_WEIGHTS` env var (default `mtg_yolo_best.pt`)
- **Training pipeline** in `model_training/`: `train.py`, `import_label_studio.py`, Roboflow + Label Studio dataset layout
- **Setup scripts** — `scripts/setup.ps1`, `scripts/setup.sh`, `scripts/check_artifacts.py`, `make check`
- **Docker Compose** — `docker compose up` with volume mounts for large artifacts
- **Docker** standalone image build (when required artifacts are present locally)

### Configuration

Environment variables (via `.env`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `YOLO_WEIGHTS` | Path to YOLO weights | `mtg_yolo_best.pt` |
| `ADMIN_KEY` | Protects `/admin/sync-set` | (unset) |

Run the API from the `app/` directory so paths to `mtg_cards.db`, `mtg_cards.index`, and `onnx_dinov2/` resolve correctly:

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Required artifacts (not all in git)

| File | In git? | Purpose |
|------|---------|---------|
| `mtg_cards.db` | Yes (root) | Card name/set lookup |
| `mtg_cards.index` | No (gitignored) | FAISS vector index |
| `onnx_dinov2/` | No (gitignored) | DINOv2 ONNX model |
| `app/mtg_yolo_best.pt` | Yes | Fine-tuned detection model |

Build index with `model_training/create_index.py` (local, gitignored). Build ONNX with `app/build_onnx.py`.

### Known limitations

- **Video size:** 100 MB upload cap; full file loaded into memory; synchronous processing (can timeout on long clips)
- **Video coverage:** Default `max_frames=300` limits how much of a long video is analyzed
- **CPU inference:** `requirements.txt` installs CPU PyTorch; training and inference are slow without CUDA
- **Duplicate cards in video JSON:** Two physical copies of the same card collapse to one unique entry
- **Web UI:** Available at `/ui` for image scan, video scan, and ZIP download
- **Fresh clone:** Run `scripts/setup.ps1` or `scripts/setup.sh`, then `make check`, before starting the API

### Known bugs

None currently tracked for Phase 1 items. Previously broken `/admin/sync-set` insert logic has been fixed.

---

## Yet to Be Implemented

Planned work is organized into phases (see internal roadmap). Summary:

### Phase 1 — Bug fixes and API polish

- [x] Fix `sync_new_cards` and persist FAISS index to disk after admin sync
- [x] Disable YOLO debug saves on `/scan` (`save_yolo=False`)
- [x] Add tunable `conf` and `dist_threshold` query params on scan endpoints

### Phase 2 — Deployment hardening

- [x] `.env.example` with documented variables
- [x] `scripts/check_artifacts.py` — validate required files before startup
- [x] Setup scripts (`setup.ps1` / `setup.sh`) for one-time environment preparation
- [x] Fix Dockerfile to include `onnx_dinov2/`; `docker-compose.yml` with volume mounts for large artifacts
- [x] Update README (YOLO11 + custom weights; correct run paths; setup docs)
- [x] Clear artifact policy in README and `.gitignore`

### Phase 3 — Simple web UI

- [x] Static frontend at `/ui` for image scan, video scan, and ZIP download
- [x] Loading states and error handling for long video scans

### Phase 4 — Tests

- [ ] Pytest unit tests for `sync_new_cards`, `TrackManager`, artifact checker
- [ ] CI smoke tests (health endpoint, optional mocked inference)

### Future — Large video support (deferred)

- [ ] Streaming uploads (chunked write to disk, not full RAM buffer)
- [ ] Redis + Celery async jobs (`POST /scan/video/async`, job polling)
- [ ] Full-video scanning without low `max_frames` cap
- [ ] Frame seek / FFmpeg-based sampling for long files
- [ ] Optional `instance_count` for duplicate physical cards in video results

### Other future enhancements

- Perspective warp / oriented bounding boxes for skewed cards
- GPU-optimized inference deployment
- Collection features (pricing, deck building) — out of scope today
- Automated index updates when new MTG sets release (after admin sync is fixed)

---

## Related Documentation

- [README.md](README.md) — How to run and test the API
- Training: `model_training/train.py`, `model_training/import_label_studio.py`
- Internal plans: video scan API, large video support, recommended improvements roadmap
