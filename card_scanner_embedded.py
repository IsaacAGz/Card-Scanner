import cv2
import numpy as np
import requests
import imutils
import os
from transformers import AutoImageProcessor, AutoModel
import torch
import faiss
import sqlite3

url = "http://192.168.100.8:8080/shot.jpg"

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

#dimension for dinov2
dimension = 768

index = faiss.read_index("mtg_cards.index")
db_conn = sqlite3.connect('mtg_cards.db')

def process_image(image):
    inputs = processor(images = image, return_tensors="pt")

    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states[:,0,:]

    return embedding.squeeze().cpu().numpy()

def order_points(points):
    rectangle = np.zeros((4,2), dtype = "float32")
    sum = points.sum(axis = 1)
    rectangle[0] = points[np.argmin(sum)]
    rectangle[2] = points[np.argmax(sum)]
    diff = np.diff(points, axis = 1)
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]

    return rectangle

def search_card(warped_image):
    rgb_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

    vector = process_image(rgb_image)

    query_vector = vector.reshape(1, -1).astype('float32')
    distances, indices = index.search(query_vector, k = 1)

    return indices[0][0], distances [0][0]

def get_card_name(faiss_id):
    cursor = db_conn.cursor()

    cursor.execute("SELECT name, set_code FROM cards WHERE faiss_id = ?", (int(faiss_id),))
    result = cursor.fetchone()

    return result[0] if result else "Unknown"

while True:
    try:
        
        image_request = requests.get(url)
        image_array = np.array(bytearray(image_request.content), dtype = np.uint8)
        frame = cv2.imdecode(image_array, -1)
        if frame is None: continue
    except Exception as e:
        print(f"connection error: {e}")
        continue
    
    frame = imutils.resize (frame, width = 600)

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale_image, (5,5), 0)
    edge_image = cv2.Canny(blurred, 75, 200)

    contours = cv2.findContours(edge_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    found_card = False
    for contour in contours:

        perimeter = cv2.arcLength(contour, True)
        epsilon = perimeter * 0.02
        approximation = cv2.approxPolyDP(contour, epsilon, True)

        if len(approximation) == 4:
            points = approximation.reshape(4,2)
            rectangle = order_points(points)
            destination_points = np.array([[0,0], [300,0], [300,420], [0,420]], dtype = "float32")
            M = cv2.getPerspectiveTransform(rectangle, destination_points)
            warped = cv2.warpPerspective(frame, M , (300, 420))

            card_idx, distances = search_card(warped)

            name = get_card_name(card_idx)
            cv2.putText(frame, f"ID: {name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # how dinov2 sees the frame
            cv2.imshow("Extracted Card", warped)
            found_card = True
            break
    
    cv2.imshow("MTG Live Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

db_conn.close()
cv2.destroyAllWindows()

