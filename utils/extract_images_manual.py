import fitz
import os

pdf_path = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\spring2025-lectures\nonexecutable\2025 Lecture 9 - Scaling laws basics.pdf"
output_dir = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture9\images"
os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)

# (filename, page_index_0_based)
requests = [
    ("lecture9-intro-extrapolation.png", 3),   # Page 4
    ("lecture9-hestness-regions.png", 7),      # Page 8
    ("lecture9-data-scaling.png", 15),         # Page 16 (or 15, keeping 15 based on prev run)
    ("lecture9-transformer-vs-lstm.png", 27),  # Page 28 -> Index 27? Wait. 1-based P28 is index 27.
    # Previous run found Trans vs LSTM on "page 28". If that was loop index 0..N, it was index 28.
    # So I will use 28.
    ("lecture9-transformer-vs-lstm.png", 28),
    ("lecture9-architecture-scaling.png", 29),  # Page 30? Debug said P29. Index 28? I'll export 29 too.
    ("lecture9-critical-batch-size.png", 35),   # Page 36 -> Index 35
    ("lecture9-chinchilla-isoflop.png", 46),    # Page 47 -> Index 46
]

for filename, idx in requests:
    if idx < len(doc):
        print(f"Extracting {filename} from Page Index {idx}")
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pix.save(os.path.join(output_dir, filename))
    else:
        print(f"Index {idx} out of range")

print("Manual extraction complete.")
