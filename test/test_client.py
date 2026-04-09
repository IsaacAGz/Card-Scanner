import requests
import cv2
import numpy as np
import time
import imutils

card_scanner_url = "http://127.0.0.1:8000/scan"
camera_url = "http://192.168.1.78:8080/shot.jpg"


def test_scan_cards():
    print("Starting Webcam Scanner... Press Ctrl+C to stop.")
    try:
        while True:
            try:
                image_request = requests.get(camera_url)

                if image_request.status_code == 200:
                    nparr = np.frombuffer(image_request.content, np.uint8)
                    display_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    display_frame = imutils.resize(display_frame, width = 900)
                    #display_frame = cv2.rotate(display_frame, cv2.ROTATE_90_CLOCKWISE)

                    _, buffer = cv2.imencode('.jpg', display_frame)

                    files = {'file': ('shot.jpg', buffer.tobytes(), 'image/jpeg')}

                    response = requests.post(card_scanner_url, files=files, timeout = 3)
                    response.raise_for_status()

                    data = response.json()
                    cards = data.get("data") or []

                    scale = 900 / 600
                    
                    for index, card in enumerate(cards):
                        raw_points = np.array(card["box"], dtype = np.int32)
                        points = (raw_points * scale).astype(np.int32)
                        name = card["name"]
                        distance = card["distance"]

                        cv2.polylines(display_frame, [points], isClosed = True, color = (0, 255,0), thickness = 2)

                        top_left_x, top_left_y = points[0]
                        label = f"{name} (distance: {distance:.2f})"
                        cv2.putText(display_frame, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        rectangle = points.astype("float32")
                        destination_points = np.array([[0,0], [420, 0], [420, 300], [0,300]], dtype = "float32")
                        M = cv2.getPerspectiveTransform(rectangle, destination_points)
                        warped_card = cv2.warpPerspective(display_frame, M, (420, 300))
                    
                        if warped_card.size > 0:
                            cv2.imshow(f"Detection Slot {index}", warped_card)
                        
                    cv2.imshow("Webcam Scanner", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print(f"API Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"\n[NETWORK ERROR] Is the Docker container running? Details: {e}")
                time.sleep(2)
                continue

    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
    
    finally:
        cv2.destroyAllWindows()
        if image_request:
            image_request.close()
        print("[INFO] Cleaned up resources and exiting.")

if __name__ == "__main__":
    test_scan_cards()