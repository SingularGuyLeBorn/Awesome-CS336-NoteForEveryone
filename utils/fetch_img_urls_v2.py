import requests
from bs4 import BeautifulSoup
import re

urls = [
    "https://arxiv.org/html/2405.04434v1",
    "https://huggingface.co/blog/speculative-decoding"
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for url in urls:
    print(f"--- Fetching {url} ---")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        soup = BeautifulSoup(response.content, 'html.parser')
        imgs = soup.find_all('img')
        for img in imgs:
            src = img.get('src')
            alt = img.get('alt', 'No alt')
            if 'figure' in src.lower() or 'fig' in src.lower() or 'img' in src.lower():
                print(f"Image: {src} | Alt: {alt}")
    except Exception as e:
        print(f"Error: {e}")
