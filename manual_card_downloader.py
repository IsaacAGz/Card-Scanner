import scrython
import time
import requests
import os

# Create the folder if it doesn't exist
folder = "BFMatcher_reference_cards"
if not os.path.exists(folder):
    os.makedirs(folder)

# List of cards you want to download for testing
cards_to_download = [
    "Cheering Crowd",
    "Reliquary Tower"
]

def download_images(card_list):
    for name in card_list:
        try:
            card = scrython.cards.Named(fuzzy=name)
            
            image_url = card.image_uris['normal']
            
            img_data = requests.get(image_url).content
            filename = f"{name.replace(' ', '_').lower()}.jpg"
            path = os.path.join(folder, filename)
            
            with open(path, 'wb') as handler:
                handler.write(img_data)
                
            print(f"Successfully downloaded: {name}")
        
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"Error downloading {name}: {e}")

download_images(cards_to_download)