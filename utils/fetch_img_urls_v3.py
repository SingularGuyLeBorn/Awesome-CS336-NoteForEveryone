import requests
from bs4 import BeautifulSoup

urls = [
    "https://arxiv.org/html/2405.04434v1", # DeepSeek V2
    "https://arxiv.org/html/2305.13245v2", # GQA
    "https://jaykmody.com/blog/speculative-sampling/" # Speculative Sampling
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

for url in urls:
    print(f"--- Fetching {url} ---")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        imgs = soup.find_all('img')
        for img in imgs:
            src = img.get('src')
            if src:
                print(f"Image: {src}")
    except Exception as e:
        print(f"Error: {e}")
