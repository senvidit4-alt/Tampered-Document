import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from core.segmenter import segment_document_targets
from core.forensics import compute_ela_map

def run_test():
    # Load genuine image
    img = cv2.imread("test_docs/genuine_marksheet.jpg")
    
    # 5 fields to tamper:
    # 1. Candidate Name (AARAV SHARMA) @ [213, 221, 311, 241] -> "VIKRAM SINGH"
    # 2. Roll No (BSE-2026-104) @ [211, 247, 297, 267] -> "BSE-2026-999"
    # 3. Mark 42 @ [699, 573, 719, 591] -> "95"
    # 4. Grade E @ [861, 576, 869, 587] -> "A"
    # 5. Total 392 @ [697, 677, 723, 695] -> "445"
    
    tamper_edits = [
        {"bbox": [205, 218, 320, 243], "text": "VIKRAM SINGH", "font_scale": 0.52, "offset": (5, 17)},
        {"bbox": [208, 245, 305, 269], "text": "BSE-2026-999", "font_scale": 0.52, "offset": (4, 17)},
        {"bbox": [695, 570, 725, 593], "text": "95", "font_scale": 0.50, "offset": (4, 16)},
        {"bbox": [855, 573, 875, 590], "text": "A", "font_scale": 0.48, "offset": (4, 14)},
        {"bbox": [693, 674, 728, 698], "text": "445", "font_scale": 0.50, "offset": (3, 16)}
    ]
    
    for edit in tamper_edits:
        x1, y1, x2, y2 = edit["bbox"]
        # 1. Flat digital whiteout box (typical of Sejda)
        img[y1:y2, x1:x2] = (255, 255, 255)
        # 2. Crisp vector-like digital text in pure black (0, 0, 0)
        cv2.putText(
            img, edit["text"],
            (x1 + edit["offset"][0], y1 + edit["offset"][1]),
            cv2.FONT_HERSHEY_SIMPLEX, edit["font_scale"], (0, 0, 0), 1, cv2.LINE_AA
        )
        
    # Re-compress uniformly at Q=85 without EXIF (mimicking web editor export)
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    
    # Save simulated image for inspection
    cv2.imwrite("outputs/simulated_sejda_tampered.jpg", recompressed)
    print("Saved outputs/simulated_sejda_tampered.jpg")

if __name__ == "__main__":
    run_test()
