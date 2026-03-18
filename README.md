========================================================================
MTG CARD RECOGNITION SYSTEM - PROJECT OVERVIEW
========================================================================

This project is a high-performance Magic: The Gathering (MTG) card 
identification system. It utilizes DINOv2 (Deep Image-based Non-parametric 
Vision) for generating image embeddings and FAISS for similarity searching 
across card records.

------------------------------------------------------------------------
1. FILE DIRECTORY
------------------------------------------------------------------------

DATA COLLECTION & DATABASE SETUP:
* auto_card_downloader.py: Bulk downloads card art from Scryfall based 
  on a JSON data file.
* manual_card_downloader.py: Utility to download specific cards by name 
  for testing.
* create_index.py: The core setup script. It processes images, generates 
  DINOv2 embeddings, builds a FAISS index, and populates a SQLite database 
  (mtg_cards.db).

SCANNER IMPLEMENTATIONS:
* card_scanner_comparison.py: A legacy version using OpenCV's ORB 
  feature matching and BFMatcher for identification.
* card_scanner_embedded.py: The primary live scanner. It captures 
  frames from an IP camera, detects card contours, and identifies cards 
  using DINOv2 and FAISS.
* card_scanner_images_embedded.py: Designed for static images. Uses 
  Roboflow/YOLO to find multiple cards in one photo and labels them 
  automatically.

------------------------------------------------------------------------
2. SYSTEM REQUIREMENTS
------------------------------------------------------------------------

HARDWARE:
* NVIDIA GPU (Highly recommended for CUDA-accelerated inference).
* Mobile Device (Used as the camera source via IP Webcam).

PYTHON DEPENDENCIES:
* torch, cv2 (opencv-python), numpy, faiss-gpu, transformers, 
  imutils, requests, scrython.

------------------------------------------------------------------------
3. TYPICAL WORKFLOW
------------------------------------------------------------------------

STEP 1: DATABASE BUILDING
1. Place a Scryfall bulk data JSON file in the root directory.
2. Run 'auto_card_downloader.py' to download reference images.
3. Run 'create_index.py' to generate 'mtg_cards.index' and 
   'mtg_cards.db'.

STEP 2: LIVE SCANNING
1. Start an MJPEG stream on your mobile device (default port 8080).
2. Run 'python card_scanner_embedded.py'.
3. The system will draw a green border around detected cards and 
   display the identified name from the database.

STEP 3: BATCH ANALYSIS
1. Place static photos in the 'images_to_scan' folder.
2. Run 'python card_scanner_images_embedded.py'.
3. Labeled results will be saved back to the same folder.

------------------------------------------------------------------------
4. TECHNICAL SPECIFICATIONS
------------------------------------------------------------------------
* Embedding Model: facebook/dinov2-base (768 dimensions).
* Search Engine: FAISS IndexFlatL2.
* Metadata Store: SQLite3.
========================================================================
