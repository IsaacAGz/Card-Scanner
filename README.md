# MTG Card Scanner

A FastAPI-based web service that detects and identifies Magic: The Gathering cards from images. It uses a Roboflow model for object detection (finding cards in an image) and a Hugging Face model (`facebook/dinov2-base`) combined with FAISS for image embeddings and similarity search to identify the specific card.

## Prerequisites

- **Python 3.11+**
- **Docker** (optional, for containerized running)
- Note: This project requires a pre-built `mtg_cards.index` (FAISS index) and `mtg_cards.db` (SQLite database) to run. These should be present in the root folder, or you can regenerate them using `create_index.py`.

## Running the Project Locally

You have two options to run the application: directly on your host machine using a Python virtual environment, or via Docker.

### Option 1: Using Docker (Recommended)

The project includes a `Makefile` that simplifies Docker commands.

1. **Build the Docker image:**
   ```bash
   make build
   ```
2. **Run the container:**
   ```bash
   make run
   ```
   The service will start in the background on port `8000`. You can verify it's running by going to `http://localhost:8000/docs` in your browser.
3. **View logs:**
   ```bash
   make logs
   ```
4. **Stop the container:**
   ```bash
   make stop
   ```

### Option 2: Using Python Virtual Environment

If you prefer to run it without Docker:

1. **Create and activate a virtual environment:**
   ```bash
   make venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the FastAPI application:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

---

## API Endpoints

### `POST /scan`
Scans an uploaded image and returns a list of detected cards with their bounding boxes and confidence distances.
- **Request Body:** Form-data with a file field named `file` (an image file like JPEG or PNG).
- **Response:** JSON containing the count of cards found and details for each card (Name, Set, Distance, Bounding Box).

---

## Testing Cases

Once the server is running (either via Docker or Uvicorn), you can test the APIs using `curl` or interactive tools.

### Test Case 1: Testing with cURL
You can test the endpoint using a sample image from the terminal. Assuming you have downloaded a test card image named `test_card.jpg` in your current directory:

```bash
curl -X POST "http://localhost:8000/scan" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_card.jpg"
```

**Example Expected Response:**
```json
{
  "count": 1,
  "cards": [
    {
      "name": "Reliquary Tower",
      "set": "con",
      "dist": 25.123,
      "box": [15, 20, 245, 340]
    }
  ]
}
```

### Test Case 2: Interactive Testing via Swagger UI
FastAPI automatically generates an interactive API documentation page.
1. Make sure the server is running.
2. Open your web browser and navigate to: [http://localhost:8000/docs](http://localhost:8000/docs).
3. Click to expand the `POST /scan` endpoint.
4. Click the **"Try it out"** button.
5. Upload a card image from your computer using the file picker.
6. Click **"Execute"**. You can view the parsed JSON result directly in your browser without any terminal commands.

---

## Additional Utilities

- **`create_index.py`**: Used to generate the FAISS index (`mtg_cards.index`) and SQLite database (`mtg_cards.db`) using Scryfall card JSON data and downloaded reference images.
- **`manual_card_downloader.py`**: A helper script to quickly download card images from Scryfall by name for testing or indexing purposes.
