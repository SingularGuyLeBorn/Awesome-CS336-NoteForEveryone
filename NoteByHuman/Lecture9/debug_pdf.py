import fitz

pdf_path = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\spring2025-lectures\nonexecutable\2025 Lecture 9 - Scaling laws basics.pdf"

doc = fitz.open(pdf_path)

keywords_to_check = [
    "Hestness", "Extrapolat", "Region", "Mixture", "Batch", "IsoFLOP", "Chinchilla"
]

for i in range(len(doc)):
    text = doc[i].get_text("text").replace('\n', ' ')
    hits = [k for k in keywords_to_check if k.lower() in text.lower()]
    if hits:
        print(f"Page {i+1} hits: {hits} | Snippet: {text[:100]}...")
