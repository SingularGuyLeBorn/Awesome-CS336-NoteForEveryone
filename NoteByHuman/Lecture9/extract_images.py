import fitz
import os

pdf_path = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\spring2025-lectures\nonexecutable\2025 Lecture 9 - Scaling laws basics.pdf"
output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture9\images"
os.makedirs(output_dir, exist_ok=True)

try:
    doc = fitz.open(pdf_path)
    print(f"Opened PDF with {len(doc)} pages.")
except Exception as e:
    print(f"Failed to open PDF: {e}")
    exit(1)

# Targets: (filename, keywords_list)
# Keywords are AND logic (all must be present)
targets = [
    # 1. Intro extrapolation
    ("lecture9-intro-extrapolation.png", ["Extrapolate", "Predictive", "Scalar"]),
    
    # 2. Hestness regions
    ("lecture9-hestness-regions.png", ["Hestness", "Power Law Region", "Irreducible"]),
    
    # 3. Data Scaling (General) - usually a log-log plot
    ("lecture9-data-scaling-log-log.png", ["Data Scaling", "Test Loss", "Dataset Size"]),
    
    # 4. Transformer vs LSTM
    ("lecture9-transformer-vs-lstm.png", ["LSTM", "Transformer", "Kaplan"]),
    
    # 5. Architecture Scaling (GLU, MoE)
    ("lecture9-architecture-scaling.png", ["Gated Linear Units", "Mixture of Experts"]),
    
    # 6. Critical Batch Size
    ("lecture9-critical-batch-size.png", ["Critical Batch Size", "Noise Scale"]),
    
    # 7. Chinchilla IsoFLOP
    ("lecture9-chinchilla-isoflop.png", ["IsoFLOP", "Chinchilla", "Hoffmann"]),
]

found = set()

for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text("text").lower()
    
    for filename, keywords in targets:
        if filename in found:
            continue
            
        match = True
        for k in keywords:
            if k.lower() not in text:
                match = False
                break
        
        if match:
            print(f"[FOUND] {filename} on Page {page_num + 1}")
            # Render high quality image
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            output_path = os.path.join(output_dir, filename)
            pix.save(output_path)
            found.add(filename)

print("Extraction finished.")
print(f"Total images found: {len(found)}")
