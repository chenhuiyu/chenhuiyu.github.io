import json
import requests
from bs4 import BeautifulSoup

data = [
        { "latLng": [52.37, 4.89], "name": "Amsterdam - Netherlands" },
        { "latLng": [37.98, 23.73], "name": "Athens - Greece" },
        { "latLng": [-8.46, 115.27], "name": "Bali - Indonesia" },
        { "latLng": [13.75, 100.49], "name": "Bangkok - Thailand" },
        { "latLng": [39.91, 116.39], "name": "Beijing - China" },
        { "latLng": [30.66, 104.06], "name": "Chengdu - China" },
        { "latLng": [29.57, 106.55], "name": "Chongqing - China" },
        { "latLng": [37.44, 116.35], "name": "Dezhou - China" },
        { "latLng": [25.29, 51.53], "name": "Doha - Qatar" },
        { "latLng": [25.07, 55.19], "name": "Dubai - United Arab Emirates" },
        { "latLng": [23.13, 113.26], "name": "Guangzhou - China" },
        { "latLng": [35.22, 139.03], "name": "Hakone - Japan" },
        { "latLng": [30.25, 120.21], "name": "Hangzhou - China" },
        { "latLng": [31.82, 117.22], "name": "Hefei - China" },
        { "latLng": [60.17, 24.94], "name": "Helsinki - Finland" },
        { "latLng": [22.28, 114.16], "name": "Hong Kong - China" },
        { "latLng": [36.65, 117.11], "name": "Jinan - China" },
        { "latLng": [1.5, 103.75], "name": "Johor Bahru - Malaysia" },
        { "latLng": [3.15, 101.69], "name": "Kuala Lumpur - Malaysia" },
        { "latLng": [34.59, 119.22], "name": "Lianyungang - China" },
        { "latLng": [32.06, 118.79], "name": "Nanjing - China" },
        { "latLng": [48.85, 2.35], "name": "Paris - France" },
        { "latLng": [36.06, 120.38], "name": "Qingdao - China" },
        { "latLng": [35.42, 119.52], "name": "Rizhao - China" },
        { "latLng": [36.41, 25.46], "name": "Santorini - Greece" },
        { "latLng": [31.23, 121.47], "name": "Shanghai - China" },
        { "latLng": [1.36, 103.82], "name": "Singapore - Singapore" },
        { "latLng": [31.3, 120.58], "name": "Suzhou - China" },
        { "latLng": [36.2, 117.08], "name": "Tai'an - China" },
        { "latLng": [39.08, 117.2], "name": "Tianjin - China" },
        { "latLng": [35.68, 139.77], "name": "Tokyo - Japan" },
        { "latLng": [48.8, 2.13], "name": "Versailles - France" },
        { "latLng": [36.71, 119.16], "name": "Weifang - China" },
        { "latLng": [37.51, 122.12], "name": "Weihai - China" },
        { "latLng": [31.49, 120.31], "name": "Wuxi - China" },
        { "latLng": [34.21, 117.28], "name": "Xuzhou - China" },
        { "latLng": [32.4, 119.41], "name": "Yangzhou - China" },
        { "latLng": [37.46, 121.44], "name": "Yantai - China" },
        { "latLng": [34.81, 117.32], "name": "Zaozhuang - China" },
        { "latLng": [36.82, 117.98], "name": "Zibo - China" }
]

access_key = "5szLk7i3AxXulJmuKqyEfw3BY5CeA7E2yQl1PnlvDsI"  # replace with your Unsplash access key

def fetch_image_url(query):
    response = requests.get(f"https://api.unsplash.com/search/photos?query={query}", headers={"Authorization": f"Client-ID {access_key}"})

    if response.status_code != 200:
        print(f"Error fetching image, status code: {response.status_code}")
        return None

    data = json.loads(response.text)
    if data["results"]:
        return data["results"][0]["urls"]["small"]  # or choose another size
    else:
        return None

for i, item in enumerate(data):
    item["frequency"] = 1
    image_url = fetch_image_url(item["name"])

    # Save the image locally
    if image_url:
        img_data = requests.get(image_url).content
        with open(f"{item['name']}-{i}.jpg", 'wb') as handler:
            handler.write(img_data)
        
    item["imageUrl"] = f"{item['name']}-{i}.jpg"