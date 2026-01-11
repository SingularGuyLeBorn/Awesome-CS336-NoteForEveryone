import requests
import os

images = [
    {
        # Mamba selection mechanism from ShumengLI/Mamba4MIS
        "url": "https://raw.githubusercontent.com/ShumengLI/Mamba4MIS/main/selection.png",
        "filename": "mamba_ssm_architecture.png"
    },
    {
        # S4 diagram from annotated-s4 images folder
        "url": "https://raw.githubusercontent.com/srush/annotated-s4/main/images/s4.png",
        "filename": "s4_architecture.png"
    },
    {
        # Alternative S4 diagram
        "url": "https://raw.githubusercontent.com/srush/annotated-s4/main/images/ssm.png",
        "filename": "s4_ssm_block.png"
    },
    {
        # MQA/GQA comparison diagram (conceptually similar to CLA's KV sharing)
        "url": "https://raw.githubusercontent.com/facebookresearch/llama/main/llama-kv-cache.webp",
        "filename": "cross_layer_attention.png"
    },
    {
        # Alternative: Unsloth DeepSeek architecture (try different path)
        "url": "https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/deepseekv3.png",
        "filename": "deepseek_v3_mla.png"
    }
]

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

for img in images:
    filepath = os.path.join(output_dir, img['filename'])
    print(f"Downloading {img['filename']}...")
    try:
        response = requests.get(img['url'], headers=headers, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  OK: {len(response.content)} bytes")
        else:
            print(f"  FAIL: Status {response.status_code}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone! Checking images folder...")
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    print(f"  {f}: {os.path.getsize(fpath)} bytes")
