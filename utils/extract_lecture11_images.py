"""
Script to extract images from Lecture 11 PDF slides.
Uses PyMuPDF (fitz) to extract embedded images and render pages as images.
"""
import fitz  # PyMuPDF
import os

# Paths
PDF_PATH = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\spring2025-lectures\nonexecutable\2025 Lecture 11 - Scaling details.pdf"
OUTPUT_DIR = r"d:\ALL IN AI\Awesome-CS336-NoteForEveryone\NoteByHuman\Lecture11\images"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_page_as_image(doc, page_num, filename, zoom=2.0):
    """Render a PDF page as a high-resolution PNG image."""
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)  # 2x zoom for higher resolution
    pix = page.get_pixmap(matrix=mat)
    output_path = os.path.join(OUTPUT_DIR, filename)
    pix.save(output_path)
    print(f"Saved: {filename}")
    return output_path

def main():
    doc = fitz.open(PDF_PATH)
    print(f"PDF has {len(doc)} pages")
    
    # Key slides to extract (0-indexed page numbers with descriptive names)
    # We'll extract specific pages that contain important figures
    slides_to_extract = [
        (0, "00_title.png"),
        (1, "01_motivation.png"),
        (2, "02_mup_learning_rate_shift.png"),
        (3, "03_cerebras_overview.png"),
        (4, "04_cerebras_scaling_comparison.png"),
        (5, "05_cerebras_mup_table.png"),
        (6, "06_cerebras_hp_search.png"),
        (7, "07_minicpm_overview.png"),
        (8, "08_minicpm_mup_params.png"),
        (9, "09_minicpm_batch_size.png"),
        (10, "10_minicpm_lr_stability.png"),
        (11, "11_wsd_schedule.png"),
        (12, "12_wsd_vs_cosine.png"),
        (13, "13_minicpm_chinchilla.png"),
        (14, "14_deepseek_overview.png"),
        (15, "15_deepseek_lr_bs_grid.png"),
        (16, "16_deepseek_scaling_fit.png"),
        (17, "17_deepseek_wsd.png"),
        (18, "18_deepseek_isoflop.png"),
        (19, "19_deepseek_prediction.png"),
        (20, "20_llama3_isoflop.png"),
        (21, "21_llama3_downstream.png"),
        (22, "22_hunyuan_isoflop.png"),
        (23, "23_minimax_scaling.png"),
        (24, "24_case_study_summary.png"),
        (25, "25_mup_intro.png"),
        (26, "26_mup_conditions.png"),
        (27, "27_mup_init_derivation.png"),
        (28, "28_mup_update_derivation.png"),
        (29, "29_mup_final_formula.png"),
        (30, "30_mup_sp_comparison.png"),
        (31, "31_lingle_overview.png"),
        (32, "32_lingle_lr_transfer.png"),
        (33, "33_lingle_activations.png"),
        (34, "34_lingle_batch_size.png"),
        (35, "35_lingle_init.png"),
        (36, "36_lingle_optimizer.png"),
        (37, "37_lingle_weight_decay.png"),
        (38, "38_lingle_10b_validation.png"),
        (39, "39_summary.png"),
    ]
    
    # Extract all pages up to the total count
    for page_num in range(min(len(doc), 45)):
        # Find the corresponding name or use default
        name = None
        for pn, fn in slides_to_extract:
            if pn == page_num:
                name = fn
                break
        if name is None:
            name = f"slide_{page_num:02d}.png"
        
        extract_page_as_image(doc, page_num, name)
    
    doc.close()
    print(f"\nDone! Extracted {min(len(doc), 45)} slides to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
