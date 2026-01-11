import requests
import os

images = [
    {
        # S4 Diagram from Hazy Research
        "url": "https://hazyresearch.stanford.edu/static/posts/2022-01-14-s4-3/s4.png", 
        "filename": "s4_architecture.png"
    },
    {
        # Mamba selection mechanism
        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mamba-ssm/selection_mechanism.png",
        "filename": "mamba_ssm_architecture.png"
    },
    {
        # CLA diagram (using a MQA diagram as proxy if strict CLA 404s, but trying a specific one first)
        "url": "https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-08_at_12.17.05_PM_2.png",
        "filename": "cross_layer_attention.png", 
        "desc": "Multi-Query/Grouped attention variant conceptualizing shared KV"
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
        response = requests.get(img['url'], headers=headers, timeout=20)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Success: {img['filename']}")
        else:
            print(f"Failed {img['filename']}: Status {response.status_code}")
    except Exception as e:
        print(f"Error {img['filename']}: {e}")
