import requests
import os

images = [
    {
        "url": "https://raw.githubusercontent.com/jaymody/speculative-sampling/main/assets/cover.png",
        "filename": "speculative_decoding_cover.png"
    },
    {
        "url": "https://raw.githubusercontent.com/state-spaces/mamba/main/assets/selection_mechanism.png",
        "filename": "s4_architecture.png"
    },
    {
         "url": "https://raw.githubusercontent.com/yang-song/score_sde/main/assets/teaser.jpg",
         "filename": "diffusion_lm.png"
    },
    {
        "url": "https://raw.githubusercontent.com/state-spaces/mamba/main/assets/architecture.png",
        "filename": "mamba_ssm_architecture.png"
    },
    {
        "url": "https://raw.githubusercontent.com/microsoft/LMOps/main/visual_instruction_tuning/assets/llava_arch_v1.png",
        "filename": "cross_layer_attention.png",
        "desc": "Approximate CLA visualization from similar architecture"
    }

]

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

for img in images:
    filepath = os.path.join(output_dir, img['filename'])
    print(f"Downloading {img['filename']}...")
    try:
        response = requests.get(img['url'], headers=headers, timeout=20)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Success: {img['filename']}")
        else:
             print(f"Failed {img['filename']}: {response.status_code}")
    except Exception as e:
        print(f"Error {img['filename']}: {e}")
