import requests
import time

card_scanner_url = "http://127.0.0.1:8000/scan"
camera_url = "http://192.168.1.84:8080/shot.jpg"

def test_scan_cards():
    print("Starting Webcam Scanner... Press Ctrl+C to stop.")
    while True:

        try:
            image_request = requests.get(camera_url)

            if image_request.status_code == 200:
            
                files = {'file': ('shot.jpg', image_request.content, 'image/jpeg')}

                response = requests.post(card_scanner_url, files=files)
                response.raise_for_status()

                data = response.json()
                
                if data["count"] > 0:
                    print(f"\n--- Found {data['count']} Cards ---")
                    for card in data["cards"]:
                        print(f"Card: {card['name']} | Set: {card['set']} | Conf: {card['dist']:.2f}")
                
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"\nConnection Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    test_scan_cards()