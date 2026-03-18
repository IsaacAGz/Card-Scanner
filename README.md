MTG Card Recognition System

This project is a high-performance Magic: The Gathering (MTG) card identification system. It utilizes DINOv2 (Deep Image-based Non-parametric Vision) for generating image embeddings and FAISS for ultra-fast similarity searching across thousands of card records.
🚀 Features

    Live Scanning: Real-time card detection and identification via mobile camera stream.

    Vector Search: Replaces traditional feature matching with deep learning embeddings for higher accuracy.

    Batch Processing: Scan and label multiple cards in static images using Roboflow/YOLO integration.

    Local Database: Uses SQLite to store card metadata linked to vector indices.

📁 File Overview
1. Data Collection & Setup

    auto_card_downloader.py: A bulk downloader that reads a Scryfall JSON bulk data file and downloads high-quality card art for the entire library.

    manual_card_downloader.py: A utility to download specific cards by name (useful for testing or small-scale hobby projects).

    create_index.py: The "Brain" of the setup. It processes all downloaded images, generates DINOv2 embeddings, builds a FAISS index, and populates the SQLite database.

2. Scanner Implementations

    card_scanner_comparison.py: The legacy/baseline version. Uses OpenCV's ORB and BFMatcher. It is slower and less robust than the embedding method but works for small sets without a GPU.

    card_scanner_embedded.py: The primary live scanner. It captures frames from an IP camera, detects the card's contour, warps the perspective, and identifies the card using DINOv2 + FAISS.

    card_scanner_images_embedded.py: Designed for static image analysis. It uses a Roboflow-hosted YOLO model to find multiple cards in one photo and labels them with their name and set code.

🛠️ Installation & Requirements
Hardware

    NVIDIA GPU (Recommended): For real-time inference using CUDA.

    Mobile Device: Used as the camera source (e.g., via IP Webcam app).

Python Dependencies
Bash

pip install torch torchvision opencv-python numpy requests imutils transformers faiss-gpu scrython

📖 How to Use
Phase 1: Building the Database

    Place your Scryfall bulk data JSON (unique-artwork...json) in the root directory.

    Run auto_card_downloader.py to populate the reference_cards folder.

    Run create_index.py. This will generate mtg_cards.index and mtg_cards.db.

Phase 2: Running the Scanner

    Ensure your phone is streaming an MJPEG feed (default URL in scripts is http://192.168.100.8:8080/shot.jpg).

    Run python card_scanner_embedded.py.

    Hold a card in front of the camera. The system will draw a green border and display the identified card name.

Phase 3: Batch Image Processing

    Place images you want to analyze in the images_to_scan folder.

    Run python card_scanner_images_embedded.py.

    The script will output new images with bounding boxes and names labeled (e.g., analyzed_photo.jpg).

📝 Technical Notes

    Model: facebook/dinov2-base (768-dimensional embeddings).

    Search: FAISS IndexFlatL2 for exact Euclidean distance matching.

    Detection: Uses cv2.findContours and approxPolyDP for 4-point card detection in live mode, and a custom YOLO model for batch mode.
