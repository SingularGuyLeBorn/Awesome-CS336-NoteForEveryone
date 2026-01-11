import requests
import os

images = [
    {
        "url": "https://raw.githubusercontent.com/omkaark/omkaark.github.io/refs/heads/main/public/7-spec-decode/draft-and-verify.png",
        "filename": "speculative_decoding_cover.png"
    },
    {
        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/temperature.png",
        "filename": "speculative_speedup.png"
    },
    {
        "url": "https://blog.vllm.ai/assets/figures/spec-decode/spec_decode_diagram.png",
        "filename": "speculative_decoding.png"
    }
]

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for img in images:
    filepath = os.path.join(output_dir, img['filename'])
    print(f"Downloading {img['filename']} from {img['url']}...")
    try:
        response = requests.get(img['url'], headers=headers, timeout=30)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Success: {img['filename']} ({len(response.content)} bytes)")
        else:
            print(f"Failed {img['filename']}: Status {response.status_code}")
    except Exception as e:
        print(f"Error {img['filename']}: {e}")
