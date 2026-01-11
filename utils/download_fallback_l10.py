import requests
from bs4 import BeautifulSoup
import os

images = [
    {
        "url": "https://jaykmody.com/blog/speculative-sampling/",
        "img_selector": "img", 
        "filename": "speculative_decoding_cover.png",
        "index": 0 
    },
    {
        "url": "https://neptune.ai/blog/state-space-models-guide",
        "img_selector": "img",
        "filename": "s4_architecture.png",
        "index": 2 
    },
    {
        "url": "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
        "img_selector": "img",
        "filename": "diffusion_lm.png",
        "index": 0
    }
]

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"

headers = {'User-Agent': 'Mozilla/5.0'}

for item in images:
    try:
        print(f"access: {item['url']}")
        r = requests.get(item['url'], headers=headers)
        soup = BeautifulSoup(r.content, 'html.parser')
        imgs = soup.find_all(item['img_selector'])
        
        if len(imgs) > item['index']:
            img_url = imgs[item['index']].get('src')
            if not img_url.startswith('http'):
                if img_url.startswith('/'):
                     # Handle relative URLs correctly
                     from urllib.parse import urljoin
                     img_url = urljoin(item['url'], img_url)
                else:
                     img_url = item['url'] + img_url

            print(f"found img: {img_url}")
            
            img_data = requests.get(img_url, headers=headers).content
            with open(os.path.join(output_dir, item['filename']), 'wb') as f:
                f.write(img_data)
            print(f"saved: {item['filename']}")
        else:
            print(f"not found index {item['index']} for {item['url']}")
    except Exception as e:
        print(f"error: {e}")
