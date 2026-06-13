import requests
import cv2
import numpy as np
import time

card_scanner_url = "http://127.0.0.1:8000/scan"
camera_url = "http://192.168.1.71:8080/shot.jpg"

def test_scan_cards():
    print("Starting Webcam Scanner... Press Ctrl+C to stop.")
    while True:
        try:
            image_request = requests.get(camera_url)

            if image_request.status_code == 200:
                nparr = np.frombuffer(image_request.content, np.uint8)
                display_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                files = {'file': ('shot.jpg', image_request.content, 'image/jpeg')}

                response = requests.post(card_scanner_url, files=files)
                response.raise_for_status()

                data = response.json()
                
                for card in data.get("cards", []):
                    x1, y1, x2, y2 = card["box"]
                    name = card["name"]
                    dist = card["dist"]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"{name} ({dist:.2f})"
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (2, 255, 0), 2)

                    crop = display_frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size > 0:
                        cv2.imshow("Card Crop", crop)
                    
                cv2.imshow("Webcam Scanner", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except requests.exceptions.RequestException as e:
            print(f"\nConnection Error: {e}")
            time.sleep(2)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_scan_cards()