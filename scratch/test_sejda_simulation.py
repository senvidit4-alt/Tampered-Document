"""
Simulate a Sejda PDF / Web editor multi-field manipulation:
5-6 fields covered with digital whiteout boxes + re-typed vector-like text,
then flattened at uniform JPEG Q=85 with stripped EXIF.
"""
import cv2
import numpy as np

def create_sejda_simulated_tamper():
    img = cv2.imread("test_docs/genuine_marksheet.jpg")
    assert img is not None, "genuine_marksheet.jpg not found"
    
    # We will modify 5 fields:
    # 1. Candidate Name area: around x=200..450, y=140..170 (AARAV SHARMA -> VIKRAMADITYA SINGH)
    # 2. Roll No area: around x=200..350, y=180..205 (BSE-2026-104 -> BSE-2026-789)
    # 3. Row 12 grade: around x=700..740, y=550..580 (Grade E -> Grade A)
    # 4. Total Marks: around x=650..750, y=650..680
    # 5. Date: around x=200..320, y=700..730
    
    # Let's inspect coordinates of OCR text in genuine_marksheet.jpg
    from core.segmenter import segment_document_targets
    segs = segment_document_targets(img)
    
    print("Found", len(segs), "segments in genuine_marksheet.jpg")
    targets_to_tamper = []
    for s in segs:
        t = s["text"].upper()
        if t in ["AARAV", "SHARMA", "104", "E", "PASS"]:
            targets_to_tamper.append(s)
            print("Target segment:", s["text"], s["tight_bbox"])

if __name__ == "__main__":
    create_sejda_simulated_tamper()
