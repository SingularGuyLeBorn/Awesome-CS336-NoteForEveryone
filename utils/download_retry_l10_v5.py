import requests
import os

# Try multiple alternative sources for each image
images = [
    # S4 - from official state-spaces repo or alternatives
    {
        "urls": [
            "https://raw.githubusercontent.com/state-spaces/s4/main/assets/s4.png",
            "https://raw.githubusercontent.com/state-spaces/s4/main/assets/overview.png",
            "https://raw.githubusercontent.com/state-spaces/s4/main/assets/s4-block.png",
            # Maarten Grootendorst blog uses substackcdn
            "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b4b6f7c-0e7f-4e9c-84a6-0bf3b1f8e41f_1600x892.png",
        ],
        "filename": "s4_architecture.png"
    },
    # CLA / MLA comparison - from Sebastian Raschka's magazine or Hugging Face
    {
        "urls": [
            "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9f4b9e8c-e1b7-4325-b18f-0b0e2dd0f7f4_1600x892.png",
            "https://mermaid.ink/img/pako:eNqFkD0OgzAMhV8lynwsDJx4CDxDlL7DAUQbUBYmRBC1qKIaH620yOAYXoYvv7-fzg1OdAYq4eCIfJYVy5Qlb5c5C0P2FNfyXK1Sl5VL0q3mPYw4u-y_74UuLI0_eHeQcMCJPEcZK4Q4zKMB21QPHQE_1LbLGLKdB1b9hfnBlGbwqCJNbVRhKbhM08oy0p5kAA==",
        ],
        "filename": "cross_layer_attention.png" 
    }
]

output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture10\images"
os.makedirs(output_dir, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

for img in images:
    filepath = os.path.join(output_dir, img['filename'])
    success = False
    
    for url in img['urls']:
        print(f"Trying {img['filename']} from {url[:60]}...")
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:
                # Check it's not an HTML error page
                content_type = response.headers.get('content-type', '')
                if 'html' not in content_type.lower():
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"  SUCCESS: {len(response.content)} bytes")
                    success = True
                    break
                else:
                    print(f"  Got HTML instead of image")
            else:
                print(f"  Failed: Status {response.status_code}, Size {len(response.content)}")
        except Exception as e:
            print(f"  Error: {e}")
    
    if not success:
        print(f"  All URLs failed for {img['filename']}")

print("\nFinal image folder contents:")
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    print(f"  {f}: {os.path.getsize(fpath)} bytes")
