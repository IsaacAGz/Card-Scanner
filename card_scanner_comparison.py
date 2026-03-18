import cv2
import numpy as np
import requests
import imutils
import os
from transformers import AutoImageProcessor, Dinov2ForImageClassification, AutoModel
import torch
from datasets import load_dataset
import faiss

url = "http://192.168.100.8:8080/shot.jpg"
orb = cv2.ORB_create(nfeatures = 1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#dimension for dinov2-base
dimension = 768

index = faiss.IndexFlatL2(dimension)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def order_points(points):
    rectangle = np.zeros((4,2), dtype = "float32")
    sum = points.sum(axis = 1)
    rectangle[0] = points[np.argmin(sum)]
    rectangle[2] = points[np.argmax(sum)]
    diff = np.diff(points, axis = 1)
    rectangle[1] = points[np.argmin(diff)]
    rectangle[3] = points[np.argmax(diff)]

    return rectangle

def process_image(image):
    inputs = processor(images = image, return_tensors="pt", device = "cuda")

    with torch.no_grad():
        outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states[:,0,:]

    return embedding.squeeze().cpu().numpy()

def search_card(vector):
    query_vector = vector.reshape(1, -1).astype('float32')

    k = 3
    
    distances, indices = index.search(query_vector, k)

    print("Nearest card indices:", indices[0])
    print("Distances (lower is better):", distances[0])

reference_data = []
ref_path = "BFMatcher_reference_cards"

try: 
    num_images = len(os.listdir(ref_path))
except FileNotFoundError:
    print(f"Error: folder {ref_path} was not found")

print(f"Indexing {num_images} cards...")
for filename in os.listdir(ref_path):
    img = cv2.imread(os.path.join(ref_path, filename), 0)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            reference_data.append({"name": filename, "descriptors": des})
print(f"Indexed {len(reference_data)} cards.")

while True:

    #Gets Frame from phone
    image_request = requests.get(url)
    #Creates array in bytes from HTTP request
    image_array = np.array(bytearray(image_request.content), dtype = np.uint8)
    #-1 for IMREAD_UNCHANGED to load image as is
    frame = cv2.imdecode(image_array, -1)
    frame = imutils.resize(frame, width = 600)

    #Detection and Warping
    #Turs image to GrayScale
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Gaussian Blur to reduce noise and keep importand features
    blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    #Edge detection using cv2 Canny
    edged_image = cv2.Canny(blurred, 75, 200)

    #Detect contours
    contours = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Return only list of contours
    contours = imutils.grab_contours(contours)
    #Sorts contours list by contour area in descending and only show the first 5
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    #cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    for contour in contours:
        
        #Detect if there is a card by examining dimension
        perimeter = cv2.arcLength(contour, True)
        epsilon = perimeter * 0.02
        approximation = cv2.approxPolyDP(contour, epsilon , True)

        #checks for 4 corners
        if len(approximation) == 4:
            
            #Orders points and warps to fill new window
            points = approximation.reshape(4,2)
            rectangle = order_points(points)
            destination_points = np.array([[0,0], [300,0], [300,420], [0,420]], dtype="float32")
            M = cv2.getPerspectiveTransform(rectangle, destination_points)
            warped = cv2.warpPerspective(frame, M, (300, 420))

            
            

            #Turn to gray scale for matching features
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            keypoints_camera, descriptors_camera = orb.detectAndCompute(warped_gray, None)

            best_match_name = "Scanning..."
            max_match = 0

            if descriptors_camera is not None:
                #Goes through each card in database to compare to card in camera view (how slow is this?)
                for card in reference_data:
            
                    matches = bf.match(descriptors_camera, card["descriptors"])

                    good = [m for m in matches if m.distance < 35]

                    if len(good) > max_match:
                        max_match = len(good)
                        best_match_name = card["name"]
                
            #Threshold for card to be a considered a match
            if max_match > 40:
                cv2.putText(frame, f"CARD: {best_match_name}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            cv2.imshow("Extracted Card", warped)
            break

    cv2.imshow("MTG Live Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
