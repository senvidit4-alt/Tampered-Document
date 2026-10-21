import sys
sys.path.insert(0, ".")
import cv2
from core.segmenter import segment_document_targets

img = cv2.imread("test_docs/genuine_marksheet.jpg")
segs = segment_document_targets(img)
for i, s in enumerate(segs):
    print(f"{i:02d}: text='{s['text']}', is_grade={s['is_grade_or_mark']}, bbox={s['tight_bbox']}")
