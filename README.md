# MTG Card Scanner

A FastAPI web service that detects and identifies Magic: The Gathering cards from images and video. It uses a fine-tuned **YOLO11** model for card detection and **DINOv2 + FAISS** for card identification against a Scryfall artwork index.

See [APPLICATION.md](APPLICATION.md) for architecture, current status, and roadmap.

## Prerequisites

- **Python 3.11+**
- **Docker** (optional, for containerized running)

### Required artifacts (not all are in git)

| Artifact | Location at runtime | In git? |
|----------|---------------------|---------|
| `mtg_yolo_best.pt` | `app/` | Yes |
| `mtg_cards.db` | `app/` | Yes (repo root copy can be synced by setup) |
| `mtg_cards.index` | `app/` | No (gitignored, ~152 MB) |
| `onnx_dinov2/` | `app/` | No (gitignored; build with `app/build_onnx.py`) |

**Artifact policy:** commit `mtg_cards.db` and `app/mtg_yolo_best.pt` if desired. Do **not** commit `mtg_cards.index` (build locally or use Git LFS / Release assets).

## First-time setup

### 1. Install dependencies and prepare artifacts

**Windows (PowerShell):**

```powershell
.\scripts\setup.ps1
```

**macOS / Linux:**

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:

- `pip install -r requirements.txt`
- Build or copy `onnx_dinov2/` into `app/`
- Copy `mtg_cards.db` and `mtg_cards.index` from repo root into `app/` when present
- Create `.env` from `.env.example` if missing

### 2. Configure environment

Copy [`.env.example`](.env.example) to `.env` and set at least:

```
YOLO_WEIGHTS=mtg_yolo_best.pt
ADMIN_KEY=your-secret-key
```

Optional video settings:

```
MAX_VIDEO_BYTES=524288000
CROP_JOB_TTL_HOURS=24
EMBEDDING_DEDUP_THRESHOLD=100
```

Optional image crop export settings:

```
MAX_IMAGE_UPLOAD_BYTES=52428800
MAX_IMAGES_PER_REQUEST=50
MAX_ZIP_INPUT_BYTES=524288000
MAX_ZIP_UNCOMPRESSED_BYTES=1073741824
```

### 3. Verify artifacts

```bash
make check
# or: python scripts/check_artifacts.py
```

### 4. Run the API

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/ui](http://localhost:8000/ui) for the web scanner, or [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

---

## Running with Docker

### Recommended: Docker Compose (mounts local artifacts)

Ensures large gitignored files are loaded from your machine without rebuilding the image each time:

```bash
make up
# or: docker compose up -d --build
```

Stop:

```bash
make down
```

Requires `mtg_cards.index`, `mtg_cards.db`, and `onnx_dinov2/` at the **repo root** (setup script can prepare these).

### Alternative: standalone Docker image

Bakes artifacts into the image (requires all files present before build):

```bash
make build
make run
make logs
make stop
```

---

## API Endpoints

### `POST /scan`
Scans an uploaded image and returns detected cards with bounding boxes.
- **Body:** form-data `file` (PNG, JPG, JPEG, WEBP)
- **Query params:** `conf` (default `0.75`), `dist_threshold` (default `300`)

### `POST /scan/video`
Scans a video and returns **unique cards** across sampled frames (track deduplication).
- **Body:** form-data `file` (MP4, MOV, AVI, MKV, WEBM)
- **Query params:** `frame_stride` (default `5`), `max_frames` (default `300`), `conf`, `dist_threshold`

### `POST /scan/video/crops`
Starts an **async job** that extracts deduplicated **camera crops** from a video (for video editing, not Scryfall art).
- **Body:** form-data `file` (MP4, MOV, AVI, MKV, WEBM)
- **Response:** `202` with `{ job_id, status, poll_url }`
- **Query params:** `sample_interval_sec` (default `5`), `max_samples` (default `0` = unlimited), `conf`, `identify` (default `false`), `dist_threshold`, `embedding_dedup_threshold` (default `100`), `track_expiry_samples` (default `3`), `no_embedding_dedup` (default `false`)

### `GET /scan/video/crops/{job_id}`
Poll job status and progress. When complete, returns `manifest`, `crop_count`, and `download_url`.

### `GET /scan/video/crops/{job_id}/download`
Download the ZIP of cropped card images + `manifest.json`. Returns `409` while the job is still running.

### `POST /scan/images/crops-zip`
Extract **camera crops** from uploaded photos or a ZIP of images (synchronous; returns the ZIP directly).
- **Body (mutually exclusive):**
  - `files` — one or more image files (PNG, JPG, JPEG, WEBP), or
  - `file` — single `.zip` archive containing images
- **Query params:** `conf` (default `0.75`), `identify` (default `false`), `dist_threshold` (default `300`)
- **Response:** `200` + `application/zip` (`image_crops.zip` with `manifest.json`)
- **Limits:** 50 MB per image, 50 images per request, 500 MB ZIP upload, 1 GB uncompressed ZIP guard (configurable via env)

### `POST /cards/images-zip`
Returns a ZIP of Scryfall `border_crop` images for a JSON list of cards.

### `POST /admin/sync-set`
Adds cards from a new Scryfall set to the index. Requires header `X-Api-Key` matching `ADMIN_KEY`.

---

## Web UI

With the API running, open [http://localhost:8000/ui](http://localhost:8000/ui):

- **Image scan** — upload a photo and view detected cards with bounding boxes
- **Extract card crops (ZIP)** (image tab) — upload multiple photos or one ZIP; downloads warped camera crops for editing
- **Video scan** — upload a clip; shows unique identified cards
- **Extract card crops (ZIP)** (video tab) — async background job; samples the video, deduplicates crops, and downloads camera footage crops for video editing
- **Download Scryfall images** — fetches Scryfall border crops for identification scan results
- **Download crops ZIP** — re-downloads camera crops from a completed crop extraction job

Advanced settings include confidence, distance threshold, optional crop identification, frame stride, sample interval, and embedding dedup threshold.

---

## Testing

### Image scan (cURL)

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "accept: application/json" \
  -F "file=@test_card.jpg"
```

### Video scan

```bash
curl -X POST "http://localhost:8000/scan/video?frame_stride=5&max_frames=300" \
  -F "file=@test_clip.mp4"
```

### Image crop extraction

```bash
# Multiple images
curl -X POST "http://localhost:8000/scan/images/crops-zip" \
  -F "files=@photo1.jpg" -F "files=@photo2.jpg" -o image_crops.zip

# ZIP of images
curl -X POST "http://localhost:8000/scan/images/crops-zip?identify=true" \
  -F "file=@binder_photos.zip" -o image_crops.zip
```

### Video crop extraction (async)

```bash
# Start job
curl -X POST "http://localhost:8000/scan/video/crops?sample_interval_sec=5" \
  -F "file=@test_clip.mp4"

# Poll status (replace JOB_ID)
curl "http://localhost:8000/scan/video/crops/JOB_ID"

# Download ZIP when complete
curl -OJ "http://localhost:8000/scan/video/crops/JOB_ID/download"
```

### Card images ZIP

```bash
curl -X POST "http://localhost:8000/cards/images-zip" \
  -H "Content-Type: application/json" \
  -o detected_cards.zip \
  -d "{\"cards\":[{\"name\":\"Reliquary Tower\",\"set\":\"con\"}]}"
```

### Test clients

```bash
python app_testing/test_client.py
python app_testing/test_video_client.py path/to/clip.mp4
python app_testing/test_video_crops_client.py path/to/clip.mp4 --output video_crops.zip
python app_testing/test_image_crops_client.py --images photo1.jpg photo2.jpg -o image_crops.zip
python app_testing/test_image_crops_client.py --zip binder_photos.zip -o image_crops.zip
python app_testing/test_image_crops_zip.py
```

---

## Training and data (local, gitignored)

Under `model_training/` (not in git):

- **`train.py`** — fine-tune YOLO11 on `dataset/`
- **`import_label_studio.py`** — import Label Studio YOLO exports
- **`create_index.py`** — build `mtg_cards.index` and `mtg_cards.db`

After training, update the app weights:

```powershell
copy model_training\runs\detect\mtg_card\weights\best.pt app\mtg_yolo_best.pt
```

---

## Makefile targets

| Target | Description |
|--------|-------------|
| `make check` | Validate required artifacts under `app/` |
| `make setup` | Run Windows setup script |
| `make up` | `docker compose up -d --build` |
| `make down` | Stop compose stack |
| `make build` / `make run` | Standalone Docker image |
