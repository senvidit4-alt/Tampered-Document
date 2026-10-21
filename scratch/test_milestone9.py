"""
Test script for Milestone 9: PDF Forensics, Check 1 Expanded Signatures, and Check 2 detectors.
"""
import io
import re
import cv2
import numpy as np
import pypdf
import pypdfium2 as pdfium
from PIL import Image

def test_pdf_parsing():
    # 1. Create a simulated Sejda-like PDF with Producer = SAMBox / Sejda
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({
        "/Producer": "Sejda PDF Desktop 7.5.3 (www.sejda.com)",
        "/Creator": "SAMBox 2.4.1"
    })
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()
    
    print("Simulated Sejda PDF size:", len(pdf_bytes))
    print("%%EOF count:", pdf_bytes.count(b"%%EOF"))
    
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    meta = reader.metadata
    print("Extracted metadata:", meta)
    
    # Test rendering via pypdfium2
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    page = pdf_doc[0]
    bitmap = page.render(scale=2.0)
    pil_img = bitmap.to_pil()
    arr = np.array(pil_img)
    print("Rendered PDF shape:", arr.shape)

if __name__ == "__main__":
    test_pdf_parsing()
