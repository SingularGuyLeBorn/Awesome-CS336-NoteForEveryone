import requests
from bs4 import BeautifulSoup
import re
import os

urls = [
    "https://blog.vllm.ai/2023/06/20/vllm.html",
    "https://arxiv.org/html/2405.04434v1"
]

for url in urls:
    print(f"--- Fetching {url} ---")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        imgs = soup.find_all('img')
        for img in imgs:
            src = img.get('src')
            alt = img.get('alt', 'No alt')
            print(f"Image: {src} | Alt: {alt}")
    except Exception as e:
        print(f"Error: {e}")
