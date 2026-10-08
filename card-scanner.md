# Card Scanner — portfolio source brief

Use this file as source material for a portfolio page. It describes the product, the stack, and the engineering experience behind it. Rewrite it in site voice. Do not invent accuracy percentages, user counts, or production-scale claims. Where this brief says something is unfinished or in-process, keep that honest.

**Project:** MTG Card Scanner  
**Builder:** Isaac Angulo  
**Span:** March 2026 – August 2026 (still evolving)  
**Kind:** Personal computer-vision product — web API, web UI, and local training pipeline  
**One line:** A service that finds Magic: The Gathering cards in photos and video, names them against official Scryfall art, and exports clean camera crops for inventory or editing.

Suggested tags: Python, FastAPI, computer vision, YOLO, DINOv2, FAISS, ONNX Runtime, OpenCV, SQLite, Docker, Magic: The Gathering.

---

## What it is

Magic cards are small, visually dense, and there are tens of thousands of them. Typing names from a pile, a binder, or a Commander game recording is slow. Card Scanner takes a photo or a video clip and does two jobs:

1. **Find** every card-shaped region in the frame.
2. **Name** each region by comparing its artwork to a local index of official card images.

It also exports the camera’s own crops — perspective-corrected rectangles of the physical cards — as a ZIP, with a JSON manifest. That path is for video editing and batch photo processing. A separate path downloads official Scryfall `border_crop` art for cards that were identified.

Typical uses I built it for:

- Scanning physical cards with a phone photo instead of typing names.
- Building a list of unique cards from a clip of a collection, trade binder, or game.
- Pulling timestamped, deduplicated card stills out of match footage.
- Batch-exporting warped crops from many binder photos, or from one ZIP of photos.
- Adding a newly released Scryfall set into the local index without rebuilding everything.

The product a person actually opens is a small web UI at `/ui`: image scan, video scan, crop extraction, and downloads. The same behavior is a FastAPI service with interactive docs at `/docs`.

---

## How it works

Identification is retrieval, not a closed-set classifier. New cards come out several times a year. A classifier would need a new class and a retrain for every printing. Nearest-neighbor search over artwork embeddings can take a new set by downloading the art, embedding it, and appending vectors.

```
Photo or video frame
        │
        ▼
YOLO11 (fine-tuned)          bounding boxes around cards
        │
        ▼
Perspective rectification    find the card quad, warp to 488×680 (about 5:7)
        │                    fall back to a padded axis-aligned crop if corners fail
        ▼
DINOv2-base (ONNX)           one embedding per crop (CLS token)
        │
        ▼
FAISS IndexFlatL2            top-10 nearest vectors in mtg_cards.index
        │
        ▼
Distance threshold           nearest hit counts only if L2 distance ≤ threshold (default 300)
        │
        ▼
SQLite                       faiss_id → name, set code, Scryfall id
```

The index is built from Scryfall `border_crop` images for paper cards. The query image is a phone or webcam crop. Warping the query toward a straight, border-cropped rectangle is what makes those two images comparable.

**Image scan** returns every detection, including ones that did not clear the distance threshold, plus the top 10 candidates and whether a warp was applied. The UI draws matched boxes in green and unmatched boxes in orange.

**Video scan** does not identify every frame. It samples with a frame stride (default every 5th frame, cap 300 processed frames), groups boxes into tracks with IoU, and identifies a track once. The response is unique `(name, set)` pairs with first-seen and last-seen timestamps and the best distance.

**Video crop export** is a background job. It seeks through the file on a time interval (default 5 seconds), warps each new card, and skips duplicates two ways: the same IoU track, and embedding distance against crops already saved. Optional identification runs in batches of 32 after the crops exist. The client polls status, then downloads a ZIP of JPEGs plus `manifest.json`.

**Image crop export** is synchronous. The client uploads many images or one ZIP. The service warps cards in each photo and returns a ZIP immediately. Identification is optional and off by default, because naming every crop is the expensive step and crop export is often just for the pictures.

**Set sync** (admin, API key, disabled when `APP_ENV=production`) asks Scryfall for a set, skips cards already in SQLite, downloads `border_crop` art, embeds the batch, appends vectors to FAISS, inserts rows whose `faiss_id` matches the new vector positions, and writes the index back to disk.

---

## Features

| Surface | What it does |
|---|---|
| `POST /scan` | One image → detections, names, sets, distances, boxes, top-10 candidates, warp flag |
| `POST /scan/video` | One video → unique identified cards across sampled frames |
| `POST /scan/video/crops` | Start an async crop job (`202` + `job_id`) |
| `GET /scan/video/crops/{job_id}` | Progress, then manifest and download URL |
| `GET /scan/video/crops/{job_id}/download` | ZIP when the job is complete (`409` while it is still running) |
| `POST /scan/images/crops-zip` | Many photos or one ZIP → warped camera crops ZIP |
| `POST /cards/images-zip` | A list of `{name, set}` → official Scryfall art ZIP |
| `POST /admin/sync-set` | Append one Scryfall set to the index and database |
| `GET /ui` | Static UI: scan, crop extract, canvas preview, ZIP downloads |
| `GET /health` | Liveness |
| CLI test clients | Webcam loop, video scan, image crops, video crops, without going through the browser |

Tunable at request time: detection confidence, identification distance, frame stride, max frames, sample interval, max samples, embedding-dedup threshold, and whether crop export should also name the cards.

---

## Tech stack

| Layer | Choice | Why it is there |
|---|---|---|
| API | FastAPI, Uvicorn, Pydantic | Upload endpoints, query params, background work via threadpool, OpenAPI docs for free |
| Detection | Ultralytics YOLO11, custom weights `mtg_yolo_best.pt` (~18 MB) | Cards in a photo are an object-detection problem. I fine-tuned YOLO11 on my own labeled card images |
| Labeling / training | Label Studio exports, Roboflow-style dataset layout, `train.py`, a training notebook | I needed a detector that fires on real photos of cards on a table, in a binder, or on a screen, not on clean Scryfall renders |
| Embeddings | `facebook/dinov2-base` | Artwork matching. OCR fails on stylized names, foils, and partial crops. A vision embedding compares the picture to the picture |
| Inference runtime | ONNX Runtime, exported with Hugging Face Optimum (`app/build_onnx.py`) | The embedding model loads once at startup. Session providers try CUDA, then CPU. The app installs CPU PyTorch wheels |
| Vector search | FAISS `IndexFlatL2`, `mtg_cards.index` (~152 MB, not in git) | Exact nearest neighbors. The index is large enough that brute-force L2 is the right first index, and simple enough to append to when a set releases |
| Metadata | SQLite `mtg_cards.db` | `faiss_id → name, set_code, scryfall_id`. The vector store and the names stay in lockstep |
| Card data | Scryfall HTTP API | Source of truth for names, set codes, and `border_crop` image URLs. Client sets a user-agent and backs off on HTTP 429 |
| Images and video | OpenCV (`opencv-python-headless`) | Decode, Canny / CLAHE / adaptive threshold, contours, perspective warp, video seek |
| Geometry | Custom `card_warp.py` | MTG cards are about 2.5×3.5 in (5:7). Warped output is 488×680 |
| Jobs | In-process `JobManager` + a worker thread | Video crop extraction outlives one HTTP request. Jobs live in temp dirs, report progress, expire after 24 hours |
| Web UI | HTML, CSS, and one `app.js` file served by FastAPI | No frontend framework. Canvas draws boxes. Crop jobs poll every 2 seconds. ZIPs download as blobs |
| Config | `python-dotenv`, `.env.example` | Weights path, admin key, upload limits, job TTL, dedup threshold, `APP_ENV` |
| Run | Dockerfile (Python 3.11-slim), Docker Compose, Makefile, `scripts/setup.ps1` and `setup.sh` | Compose bind-mounts the heavy artifacts so they are not baked into every image rebuild |
| Checks | `scripts/check_artifacts.py`, `make check` | A fresh clone fails clearly when the index or ONNX model is missing |

Python dependencies that matter: `fastapi`, `uvicorn`, `python-multipart`, `opencv-python-headless`, `numpy`, CPU `torch` / `torchvision`, `transformers`, `faiss-cpu`, `ultralytics`, `python-dotenv`, `optimum[onnxruntime]`, `onnxruntime`.

---

## Architecture, in practice

```
Card Scanner/
├── app/
│   ├── main.py               # FastAPI app, lifespan model load, scan and admin routes
│   ├── video_scan.py         # IoU TrackManager, frame-stride video identification
│   ├── video_crops.py        # Seek-based sampling, track + embedding dedup, ZIP
│   ├── image_crops.py        # Multi-image and ZIP input, path and zip-bomb checks
│   ├── card_warp.py          # Quad finding, aspect scoring, perspective warp
│   ├── crop_export.py        # JPEG encode, manifest entries, ZIP builder
│   ├── crop_job_worker.py    # Background crop job
│   ├── job_manager.py        # Thread-safe in-process job store
│   ├── inference_runtime.py  # Same ONNX + FAISS stack for CLI tools
│   ├── card_images.py        # Scryfall image ZIP
│   ├── build_onnx.py         # One-time DINOv2 → ONNX
│   └── static/               # index.html, app.js, style.css
├── app_testing/              # HTTP clients and in-process crop runners
├── scripts/                  # setup, artifact check, warp debug previews
├── model_training/           # Local only (gitignored): dataset, train, index build
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

Models load once in the FastAPI lifespan: YOLO weights, ONNX session, image processor, FAISS index, SQLite connection (`check_same_thread=False`). Inference runs in a threadpool so the event loop stays free. Heavy files are an explicit policy: commit the SQLite DB and the YOLO weights if you want; do not commit the FAISS index or `onnx_dinov2/`. Setup copies or builds them into `app/` for local runs. Compose mounts the copies that live at the repo root.

Upload limits are environment variables. Video defaults to a few hundred megabytes. Image crop requests cap per-file size, image count, ZIP upload size, and uncompressed ZIP size. ZIP extraction rejects absolute paths, `..`, drive-letter paths, `__MACOSX`, and dotfiles.

---

## What I actually built, in order

The git history is the experience. It started as a recognition write-up and became a service I could point a browser at.

**March 2026 — the idea and the first API.**  
I wrote the project up as an MTG card recognition system, then stood up a FastAPI app and a Dockerfile. The data side was a SQLite catalog plus a FAISS index of card art. A small webcam client posted frames at the scan endpoint so I could see detections live.

**April–May 2026 — detection had to be learned, not assumed.**  
Early image handling was not good enough on real photos. I switched the detector to a pretrained YOLO11 model and changed the pipeline so each prediction is cropped before it is embedded. Training lived in a notebook, then in a local `model_training/` folder: import Label Studio YOLO exports, fine-tune on my dataset, copy `best.pt` into the app as `mtg_yolo_best.pt`. That was the point where “find the rectangle” stopped being a hand-tuned OpenCV hack and became a model I could retrain when it missed cards.

**June 2026 — the index became something I could operate.**  
I split the app into an `app/` package, added a health check, and built `POST /admin/sync-set`. Sync walks Scryfall search pagination, keeps paper cards that have a `border_crop`, skips rows already stored, embeds new art in one batch, and appends to FAISS. I exported DINOv2 to ONNX with Optimum and pointed both the API and the index builder at ONNX Runtime, so indexing and serving use the same embedding function. The index builder can resume, because rebuilding every card from scratch is too slow to do casually. I also had to fix the sync insert so each new row’s `faiss_id` is the position FAISS actually assigned, then persist the index to disk. A vector append that is not committed, or a row whose id does not match the vector, silently identifies the wrong card.

**July 2026 — video, crops, and a UI.**  
Image scan was not the workflow I wanted for a pile of cards or a recorded game. I added video scanning with stride, a frame cap, and IoU tracking so a card that sits on camera is one result, not one result per frame. Crop export came next, because sometimes the output I want is the picture, not the name. Video crops run as jobs: write the upload to disk, sample by timestamp, save JPEGs, poll, download. Image crops stay synchronous and accept either many files or a ZIP. The UI grew up beside the API: tabs, advanced settings, loading and error states, a canvas preview, Scryfall ZIP download, and crop ZIP download. Docker Compose, setup scripts, and an artifact checker landed in the same stretch so the service was runnable on a machine that did not already have my notebook state.

**August 2026 — making identification inspectable, then making crops straight.**  
Phase 0 of accuracy work changed the match from “top-1 or nothing” to a top-10 candidate list, and attached a `warped` flag to each detection. I needed to see whether a miss was a bad neighbor, a threshold that was too tight, or a crop that was never straightened. There is no published accuracy number from that pass. The machinery to measure is what shipped.

Right after that I reworked perspective correction. A single Canny pass loses the card border under glare and uneven room light. The warp module now builds several edge maps — adaptive threshold plus Canny, CLAHE plus Canny, and plain Canny — closes the edges, and collects convex quads. Each quad is scored on two things: how close its area is to “most of the crop, but not the whole padded box,” and how close its aspect is to a real card (height/width near 680/488, rejected outside a wide band). The best quad is warped with `getPerspectiveTransform`. If the quad is missing or the warp is tiny, the pipeline keeps the padded crop and says `warped: false`. I added `scripts/debug_warp.py` to dump the box, the chosen quad, and the rectified image so I could judge this by looking, not by guessing.

---

## Experience I can stand behind

These are the judgments the project forced, not a list of libraries I imported.

**Open-set recognition is a search problem.**  
I treated “which card is this?” as nearest-neighbor retrieval over artwork. The catalog grows by appending embeddings and rows. Distance is a first-class output. A detection can be real and still unidentified. The UI shows that difference instead of hiding misses inside a single label.

**The crop is the product.**  
YOLO’s box is axis-aligned. Cards in hand, in sleeves, and on a table are not. If I embed the raw box, the vector is full of table, fingers, and perspective. I pad the box, hunt for the card quad, and warp to a canonical 5:7 frame that resembles the Scryfall art the index was built on. Warp failure is allowed. A bad warp is worse than no warp, so the fallback is explicit.

**Glare is a reason to run more than one edge detector.**  
Specular highlights on sleeves delete the border in one threshold and leave it in another. I stopped looking for one magic Canny pair and started scoring candidates across a few illuminations (adaptive threshold, CLAHE, plain blur). The score prefers a quad that is card-shaped and fills the detection without swallowing the padding.

**Video has two different dedup problems.**  
While a card stays in frame, IoU tracking is enough and it is cheap. When the camera cuts, the hand moves, or the same printing shows up later, the track id is new and the artwork is not. Embedding distance catches that second case. I learned to keep the threshold configurable, because a tight threshold keeps two physical copies and a loose one merges them. For a game VOD I usually want one still per printing. For inventory of duplicates, that merge is the wrong behavior, and the API can turn embedding dedup off.

**Long work does not belong inside one request.**  
Synchronous video scan is capped (`max_frames`, default 300) because CPU inference on every sampled frame will blow a proxy timeout. Crop extraction of a long file is a job: `202`, poll, `409` if you download early, ZIP when it finishes, delete the upload in a `finally`, expire the job directory after a TTL. I kept image batches synchronous on purpose. They have a hard image cap and a size cap, and the client can wait. I did not reach for Celery. An in-process store was the honest scope. It will not survive a restart, and I know that.

**Same embedding function on both sides of the index.**  
Index build and query both go through DINOv2’s CLS token via ONNX. I exported once with Optimum and load the processor from the same folder. Mixing a PyTorch embed at index time with an ONNX embed at query time would make distances meaningless. Batching matters too: set sync embeds a whole set at once; video identification only embeds tracks that do not already have a name; crop identification runs in batches of 32 after files are on disk.

**Operational details that bit me.**  
FAISS row ids and SQLite `faiss_id` must be the same integer. The index file has to be rewritten after an append or the next process starts from the old catalog. Scryfall will 429 a tight loop, so sync sleeps and retries. Large artifacts do not belong in git the same way source does; the README states that, setup scripts place them, and Compose mounts them. ZIP upload is an attack surface: I check entry names and uncompressed size before trusting the archive. Admin sync is key-protected and returns 404 in production so the mutation endpoint is not part of the public surface.

**A thin UI was the right UI.**  
I served three static files from the same process as the model. That kept one deployable, one port, and one place to read errors. The interesting client behavior is small and specific: disable every action while a scan runs, poll a crop job without blocking the tab, draw boxes on a canvas in the image’s own coordinates, and hand the browser a ZIP blob. I did not need a component framework to learn whether the pipeline was right.

**I debugged vision by saving pictures.**  
`scripts/debug_warp.py` and the candidate list exist because a wrong card name does not tell you whether detection, warp, or the neighbor search failed. The `warped` flag and the top-10 list are there so the next accuracy pass can separate those failures.

---

## Skills this project is evidence of

- Designing a multi-stage vision pipeline: detect, rectify, embed, search, explain the miss.
- Fine-tuning an object detector on a custom dataset and wiring the weights into a service.
- Using a foundation vision model as a feature extractor, including ONNX export and a shared runtime for indexing and serving.
- Building and incrementally updating a vector index, with a relational table as the source of names.
- Classical OpenCV when the learned box is not the final geometry: edges, morphology, contour approximation, perspective transforms, aspect constraints.
- Video processing choices: stride vs. timestamp seek, IoU tracking, embedding dedup, memory and timeout limits.
- API design for mixed workloads: sync JSON, sync ZIP, async job with poll and download.
- Defensive upload handling and environment-driven limits.
- A deploy story that respects artifact size: Docker, Compose volume mounts, setup scripts, a preflight check.
- A small accessible UI (tabs, labels, canvas, progress) on top of that API.

---

## Honest limits

Say these if the page has a “what I’d do next” or “constraints” note. Do not imply they are already solved.

- Inference in this repo is CPU-first. CUDA is a provider fallback, not a tuned deployment.
- Video uploads are buffered up to `MAX_VIDEO_BYTES`. There is no chunked streaming upload.
- Crop jobs are in-process. A restart drops them. There is no Redis or Celery.
- Synchronous video scan will not walk an entire long file; the frame cap is intentional.
- Image crop export does not dedupe the same card across two photos.
- Two physical copies of one printing collapse to one unique card in video JSON, and embedding dedup can do the same to crops.
- Perspective warp is heuristic. Sleeves, glare, and busy backgrounds still miss the quad; the axis-aligned fallback covers that and is labeled.
- Accuracy measurement was started (top-k candidates, warp flag, warp debug dumps). I do not have a final precision/recall figure to publish.
- Pytest coverage and CI smoke tests are still planned, not the safety net.
- Collection features — prices, decks, collection tracking — are out of scope.

---

## Suggested portfolio framing

**Title:** MTG Card Scanner

**Subtitle:** Detect and identify Magic cards in photos and video, and export perspective-corrected crops.

**Short blurb (about 50 words):**  
I built a FastAPI service that finds Magic: The Gathering cards with a fine-tuned YOLO11 model, straightens them with OpenCV, and names them by searching DINOv2 embeddings in a FAISS index of Scryfall art. It scans stills and video, deduplicates cards across a clip, and exports crops for editing or inventory.

**Three bullets if the layout is tight:**

- Fine-tuned YOLO11 for detection, DINOv2 on ONNX Runtime for artwork embeddings, FAISS plus SQLite for open-set identification that can absorb a new Scryfall set.
- Perspective warp aimed at real photos: multi-strategy edges, card-aspect scoring, and a safe fallback when glare hides the corners.
- Video treated as a systems problem: IoU tracks, embedding dedup, frame caps on the sync API, and background jobs with progress and ZIP download for long clips.

**A longer “what I learned” pull-quote, if the page wants one:**  
The card’s name was never the hard part. The hard part was getting a crooked, glare-covered photo into the same visual space as the official art, and then deciding when a video frame was a new card.
