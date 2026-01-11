import requests
import os

# Define images and their sources
images = [
    {
        "filename": "speculative_decoding_cover.png",
        "url": "https://raw.githubusercontent.com/jaymody/speculative-sampling/main/assets/cover.png",
        "desc": "Speculative Sampling Diagram from Jay Mody's repo"
    },
    {
        "filename": "mha_gqa_mla_comparison.svg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/2f/DeepSeek_KV_cache_comparison_between_MHA%2C_GQA%2C_MQA%2C_MLA.svg",
        "desc": "DeepSeek MLA comparison SVG"
    },
    {
        "filename": "paged_attention.gif",
        "url": "https://blog.vllm.ai/assets/figures/annimation0.gif",
        "desc": "vLLM PagedAttention Animation"
    }
]

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
os.makedirs(output_dir, exist_ok=True)

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://google.com'
})

print(f"Downloading images to {output_dir}...")

for img in images:
    filepath = os.path.join(output_dir, img["filename"])
    print(f"\nTarget: {img['filename']}")
    print(f"Source: {img['url']}")
    
    try:
        response = session.get(img['url'], timeout=30)
        if response.status_code == 200:
            content_size = len(response.content)
            print(f"Status: 200 OK, Size: {content_size} bytes")
            
            if content_size < 1000:
                print("WARNING: File too small, likely an error page or empty.")
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print("Saved successfully.")
        else:
            print(f"FAILED: Status Code {response.status_code}")
    except Exception as e:
        print(f"ERROR: {e}")
