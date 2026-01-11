import requests
import os

# Define images and their target filenames
images = [
    {
        "url": "https://raw.githubusercontent.com/jaymody/speculative-sampling/main/assets/cover.png",
        "filename": "speculative_decoding_cover.png"
    },
    {
        "url": "https://production-media.paperswithcode.com/methods/Screen_Shot_2022-07-16_at_12.58.58_PM.png", 
        "filename": "cross_layer_attention.png",
        "desc": "Cross Layer Attention schematic"
    },
    {
        "url": "https://production-media.paperswithcode.com/methods/Screen_Shot_2022-01-26_at_11.53.25_AM.png",
        "filename": "s4_architecture.png",
        "desc": "Structured State Space (S4) model diagram"
    },
    {
        "url": "https://production-media.paperswithcode.com/methods/Screen_Shot_2022-05-30_at_2.51.52_PM.png",
        "filename": "diffusion_lm.png",
        "desc": "Diffusion-LM generation process"
    },
    # Mamba associative recall weakness is conceptual, using a general Mamba/SSM comparison or related figure
    {
        "url": "https://production-media.paperswithcode.com/methods/mamba_arch_2.png",
        "filename": "mamba_ssm_architecture.png", 
        "desc": "Mamba vs Transformer architecture"
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
            print(f"Failed to download {img['filename']}: Status {response.status_code}")
    except Exception as e:
        print(f"Error downloading {img['filename']}: {e}")
