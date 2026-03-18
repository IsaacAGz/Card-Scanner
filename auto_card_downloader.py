import json
import requests
import time
import os

with open("unique-artwork-20260314210622.json", "r", encoding = "utf-8") as f:
    cards = json.load(f)

output_dir = "reference_cards"
os.makedirs(output_dir, exist_ok=True)

print(f"Total cards to Download: {len(cards)}")

for i, card in enumerate(cards):
    if 'image_uris' in card and 'normal' in card['image_uris'] and 'paper' in card.get('games', []):

        file_path = os.path.join(output_dir, f"{card['id']}.jpg")

        if not os.path.exists(file_path):
            try:
                response = requests.get(card['image_uris']['normal'], timeout = 10)
                if response.status_code == 200:
                    with open(file_path, 'wb') as handler:
                        handler.write(response.content)

                    time.sleep(0.1)
                else:
                    print(f"Failed to download {card['id']}: Status {response.status_code}")
            except Exception as e:
                print(f"Error downloading {card['id']}: {e}")
        if i % 100 == 0:
                print(f"Downloaded: {i}/{len(cards)}...")

    
