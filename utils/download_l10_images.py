import requests
import os

images = {
    "mha_gqa_mla_comparison.svg": "https://upload.wikimedia.org/wikipedia/commons/2/2f/DeepSeek_KV_cache_comparison_between_MHA%2C_GQA%2C_MQA%2C_MLA.svg",
    "paged_attention.gif": "https://blog.vllm.ai/assets/figures/annimation0.gif",
    "speculative_decoding.png": "https://i.imgur.com/YrLebkI.png",
    "speculative_speedup.png": "https://i.imgur.com/rhR3U46.png"
}

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for filename, url in images.items():
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        with open(os.path.join(output_dir, filename), 'wb') as f:
            f.write(response.content)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
