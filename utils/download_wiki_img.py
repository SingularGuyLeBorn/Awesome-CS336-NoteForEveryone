import requests
from bs4 import BeautifulSoup
import os

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0'
})

# Helper to find image on Wikimedia Commons page
def download_wikimedia_image(page_url, filename):
    print(f"Fetching page: {page_url}")
    try:
        r = session.get(page_url)
        if r.status_code != 200:
            print(f"Page failed: {r.status_code}")
            return False
            
        soup = BeautifulSoup(r.content, 'html.parser')
        # Look for the "Original file" link or the main image display
        # Often class="internal" or the fullImageLink div
        div = soup.find('div', class_='fullImageLink')
        if div:
            a = div.find('a')
            if a:
                img_url = a.get('href')
                if not img_url.startswith('http'):
                    img_url = 'https:' + img_url
                print(f"Found Image URL: {img_url}")
                
                # Download
                r_img = session.get(img_url)
                if r_img.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(r_img.content)
                    print(f"Saved {filename} ({len(r_img.content)} bytes)")
                    return True
        print("Could not find image link on page.")
        # Debug: check title
        print(f"Page Title: {soup.title.string if soup.title else 'No Title'}")
    except Exception as e:
        print(f"Error: {e}")
    return False

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
img_path = os.path.join(output_dir, "mha_gqa_mla_comparison.svg")

# Try 1: The KV cache comparison file page
success = download_wikimedia_image("https://commons.wikimedia.org/wiki/File:DeepSeek_KV_cache_comparison_between_MHA,_GQA,_MQA,_MLA.svg", img_path)

if not success:
    # Try 2: The MoE and MLA file page
    print("Trying alternative file page...")
    download_wikimedia_image("https://commons.wikimedia.org/wiki/File:DeepSeek_MoE_and_MLA_(DeepSeek-V2).svg", img_path)
