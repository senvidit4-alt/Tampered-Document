"""
DocTamperGuard - Milestone 7A: Five-Check Verification Dataset Generator
Generates ground-truth benchmark documents covering all 5 core document fraud types:
1. Type 1: Edited Genuine Marksheet (Photoshop metadata, ELA spike, QR mismatch)
2. Type 2: Fabricated Certificate from Real University (Fake QR signature, Non-existent Roll)
3. Type 3: Unrecognised Institution (Blacklisted/Unrecognised by UGC)
4. Type 4: Fully AI-Generated Document (No QR, AI metadata, non-standard layout)
5. Type 5: Genuine Document, Wrong Person (Tested using genuine_marksheet with mismatched applicant)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode

# Setup directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DOCS_DIR = PROJECT_ROOT / "test_docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TEST_DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size: int, bold: bool = False):
    """
    Load high-quality Windows TrueType fonts with graceful fallback.
    """
    candidates = (
        ["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf", "timesbd.ttf"]
        if bold
        else ["arial.ttf", "calibri.ttf", "segoeui.ttf", "times.ttf"]
    )
    for name in candidates:
        font_path = Path("C:/Windows/Fonts") / name
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                pass
    return ImageFont.load_default()


def create_scanned_canvas(width: int = 1000, height: int = 1200, seed: int = 42) -> Image.Image:
    """
    Creates a realistic off-white document canvas with subtle paper grain/noise
    and scan illumination gradient.
    """
    np.random.seed(seed)
    base_color = np.array([250.0, 248.0, 244.0], dtype=np.float32)
    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color

    # Paper grain noise
    noise = np.random.normal(0.0, 3.5, (height, width, 3))

    # Scanner radial illumination vignette
    y_grid = np.linspace(-1.0, 1.0, height)[:, None]
    x_grid = np.linspace(-1.0, 1.0, width)[None, :]
    vignette = 1.0 - 0.025 * (x_grid**2 + y_grid**2)

    canvas = (canvas + noise) * vignette[:, :, None]
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def make_qr_code_image(payload: Dict, box_size: int = 3, border: int = 2) -> Image.Image:
    """
    Generates a scannable QR Code image encoding a JSON payload.
    """
    qr = qrcode.QRCode(box_size=box_size, border=border)
    qr.add_data(json.dumps(payload))
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def draw_marksheet(
    img: Image.Image,
    issuer_title: str,
    subtitle: str,
    candidate_name: str,
    roll_no: str,
    dob: str,
    institution: str,
    center_code: str,
    date_of_issue: str,
    subjects: List[Tuple[str, str, str, str, str]],
    qr_payload: Optional[Dict] = None
) -> Dict:
    """
    Draws a structured, high-resolution marksheet on the canvas.
    Returns coordinates of the 4th subject grade cell for tampering simulation.
    """
    draw = ImageDraw.Draw(img)

    NAVY_PRIMARY = (20, 35, 65)
    SLATE_DARK = (45, 55, 72)
    BORDER_COLOR = (70, 80, 95)
    BORDER_GOLD = (180, 150, 90)
    TABLE_HEADER_BG = (234, 238, 244)
    TEXT_MUTED = (95, 105, 120)
    SEAL_COLOR = (175, 40, 40)

    # 1. Ornamental Double Border
    draw.rectangle([(25, 25), (975, 1175)], outline=BORDER_COLOR, width=3)
    draw.rectangle([(32, 32), (968, 1168)], outline=BORDER_GOLD, width=1)
    for cx, cy in [(32, 32), (968, 32), (32, 1168), (968, 1168)]:
        draw.rectangle([(cx - 4, cy - 4), (cx + 4, cy + 4)], fill=BORDER_GOLD)

    # 2. Header Section
    font_title = get_font(22, bold=True)
    font_subtitle = get_font(13, bold=False)
    font_session = get_font(12, bold=True)

    seal_center = (500, 75)
    draw.ellipse([(seal_center[0] - 28, seal_center[1] - 28),
                  (seal_center[0] + 28, seal_center[1] + 28)],
                 outline=NAVY_PRIMARY, width=2)
    draw.ellipse([(seal_center[0] - 24, seal_center[1] - 24),
                  (seal_center[0] + 24, seal_center[1] + 24)],
                 outline=BORDER_GOLD, width=1)
    draw.text((seal_center[0] - 14, seal_center[1] - 10), "★ BSE ★",
              fill=NAVY_PRIMARY, font=get_font(7, bold=True))

    draw.text((500, 120), issuer_title, fill=NAVY_PRIMARY, font=font_title, anchor="mm")
    draw.text((500, 148), subtitle, fill=SLATE_DARK, font=font_subtitle, anchor="mm")
    draw.text((500, 170), "ACADEMIC YEAR: 2025 - 2026  |  ANNUAL PERFORMANCE STATEMENT",
              fill=TEXT_MUTED, font=font_session, anchor="mm")

    draw.line([(60, 190), (940, 190)], fill=BORDER_COLOR, width=1)

    # 3. Candidate Information Card
    draw.rectangle([(60, 205), (940, 310)], fill=(245, 246, 250), outline=(205, 212, 224), width=1)

    font_label = get_font(12, bold=True)
    font_val = get_font(12, bold=False)

    cand_left = [
        ("Candidate Name:", candidate_name),
        ("Roll Number:", roll_no),
        ("Date of Birth:", dob),
    ]
    cand_right = [
        ("Institution:", institution),
        ("Center Code:", center_code),
        ("Date of Issue:", date_of_issue),
    ]

    y_pos = 225
    for lbl, val in cand_left:
        draw.text((80, y_pos), lbl, fill=NAVY_PRIMARY, font=font_label)
        draw.text((215, y_pos), val, fill=SLATE_DARK, font=font_val)
        y_pos += 26

    y_pos = 225
    for lbl, val in cand_right:
        draw.text((530, y_pos), lbl, fill=NAVY_PRIMARY, font=font_label)
        draw.text((645, y_pos), val, fill=SLATE_DARK, font=font_val)
        y_pos += 26

    # 4. Structured Subject Marks Table
    table_top = 345
    col_x = [60, 160, 480, 630, 790, 940]
    row_h = 55
    num_subjects = len(subjects)

    draw.rectangle([(col_x[0], table_top), (col_x[-1], table_top + 45)],
                   fill=TABLE_HEADER_BG, outline=BORDER_COLOR, width=1)

    headers = ["CODE", "SUBJECT TITLE", "MAX MARKS", "MARKS OBT.", "GRADE"]
    font_th = get_font(13, bold=True)
    for i in range(len(headers)):
        cx = (col_x[i] + col_x[i + 1]) // 2
        draw.text((cx, table_top + 22), headers[i], fill=NAVY_PRIMARY, font=font_th, anchor="mm")

    font_td = get_font(13, bold=False)
    font_grade = get_font(15, bold=True)
    grade_4_coords = {}

    for idx, (code, name, max_m, obt_m, grade) in enumerate(subjects):
        ry = table_top + 45 + idx * row_h
        if idx % 2 == 1:
            draw.rectangle([(col_x[0], ry), (col_x[-1], ry + row_h)], fill=(247, 248, 252))

        draw.rectangle([(col_x[0], ry), (col_x[-1], ry + row_h)], outline=(210, 215, 225), width=1)

        draw.text(((col_x[0] + col_x[1]) // 2, ry + row_h // 2), code, fill=SLATE_DARK, font=font_td, anchor="mm")
        draw.text((col_x[1] + 25, ry + row_h // 2), name, fill=SLATE_DARK, font=font_td, anchor="lm")
        draw.text(((col_x[2] + col_x[3]) // 2, ry + row_h // 2), max_m, fill=SLATE_DARK, font=font_td, anchor="mm")
        draw.text(((col_x[3] + col_x[4]) // 2, ry + row_h // 2), obt_m, fill=SLATE_DARK, font=font_td, anchor="mm")

        grade_x = (col_x[4] + col_x[5]) // 2
        grade_y = ry + row_h // 2
        draw.text((grade_x, grade_y), grade, fill=NAVY_PRIMARY, font=font_grade, anchor="mm")

        if idx == 3:
            grade_4_coords = {
                "cell_bbox": (col_x[4], ry, col_x[5], ry + row_h),
                "center": (grade_x, grade_y),
                "patch_bbox": (grade_x - 35, grade_y - 20, grade_x + 35, grade_y + 20),
            }

    # Summary Row
    summary_y = table_top + 45 + num_subjects * row_h
    draw.rectangle([(col_x[0], summary_y), (col_x[-1], summary_y + 45)],
                   fill=(240, 243, 248), outline=BORDER_COLOR, width=1)
    font_summary = get_font(12, bold=True)
    draw.text((col_x[1] + 25, summary_y + 22), "TOTAL MARKS / RESULT SUMMARY", fill=NAVY_PRIMARY, font=font_summary, anchor="lm")
    
    tot_max = sum(int(s[2]) for s in subjects)
    tot_obt = sum(int(s[3]) for s in subjects)
    draw.text(((col_x[2] + col_x[3]) // 2, summary_y + 22), str(tot_max), fill=NAVY_PRIMARY, font=font_summary, anchor="mm")
    draw.text(((col_x[3] + col_x[4]) // 2, summary_y + 22), str(tot_obt), fill=NAVY_PRIMARY, font=font_summary, anchor="mm")
    draw.text(((col_x[4] + col_x[5]) // 2, summary_y + 22), "PASS", fill=NAVY_PRIMARY, font=font_summary, anchor="mm")

    total_table_height = 45 + num_subjects * row_h + 45
    for cx in col_x:
        draw.line([(cx, table_top), (cx, table_top + total_table_height)], fill=BORDER_COLOR, width=1)

    # 5. Security & Verification Bar
    status_box_y = summary_y + 65
    draw.rectangle([(60, status_box_y), (940, status_box_y + 36)],
                   fill=(248, 250, 252), outline=(195, 205, 220), width=1)
    font_status = get_font(11, bold=True)
    draw.text((500, status_box_y + 18),
              "RESULT: PASSED SECONDARY STANDING  |  CENTRAL ARCHIVE HASH: 9a7b-33f2-bc10-449e-d81a",
              fill=NAVY_PRIMARY, font=font_status, anchor="mm")

    # 6. Embed Real Machine-Readable QR Code (if provided)
    if qr_payload:
        qr_img = make_qr_code_image(qr_payload, box_size=3, border=2)
        # Position in bottom-right above controller signature: x=750, y=825
        qr_x, qr_y = 750, 825
        img.paste(qr_img, (qr_x, qr_y))

        # QR border and label
        draw.rectangle([(qr_x - 3, qr_y - 3), (qr_x + qr_img.size[0] + 3, qr_y + qr_img.size[1] + 3)],
                       outline=BORDER_COLOR, width=1)
        draw.text((qr_x + qr_img.size[0] // 2, qr_y + qr_img.size[1] + 10),
                  "SCAN FOR CENTRAL VERIFICATION",
                  fill=TEXT_MUTED, font=get_font(7, bold=True), anchor="mm")

    # 7. Official Stamp & Signatures
    stamp_x, stamp_y = 500, 960
    draw.ellipse([(stamp_x - 52, stamp_y - 52), (stamp_x + 52, stamp_y + 52)],
                 outline=SEAL_COLOR, width=2)
    draw.ellipse([(stamp_x - 45, stamp_y - 45), (stamp_x + 45, stamp_y + 45)],
                 outline=SEAL_COLOR, width=1)
    draw.text((stamp_x, stamp_y - 22), "BOARD OF SECONDARY ED.", fill=SEAL_COLOR, font=get_font(7, bold=True), anchor="mm")
    draw.text((stamp_x, stamp_y), "★ OFFICIAL SEAL ★", fill=SEAL_COLOR, font=get_font(8, bold=True), anchor="mm")
    draw.text((stamp_x, stamp_y + 22), "EXAMINATION DEPT", fill=SEAL_COLOR, font=get_font(7, bold=True), anchor="mm")

    draw.line([(90, 980), (310, 980)], fill=BORDER_COLOR, width=1)
    draw.text((200, 995), "Signature of Candidate", fill=SLATE_DARK, font=get_font(11, bold=False), anchor="mm")

    # Controller signature
    sig_points = [(690, 970), (710, 955), (730, 975), (750, 950), (780, 970), (810, 958), (840, 965)]
    draw.line(sig_points, fill=(15, 25, 80), width=2)
    draw.line([(670, 980), (890, 980)], fill=BORDER_COLOR, width=1)
    draw.text((780, 995), "Controller of Examinations", fill=SLATE_DARK, font=get_font(11, bold=True), anchor="mm")

    draw.text((500, 1145),
              "SECURITY PROTECTED DOCUMENT • ANY ALTERATION OR TAMPERING VOIDS THIS STATEMENT OF MARKS",
              fill=(130, 140, 155), font=get_font(9, bold=True), anchor="mm")

    return grade_4_coords


def generate_benchmark_suite():
    """
    Generates all benchmark test files in test_docs/.
    """
    print("=========================================================")
    print("  DocTamperGuard - Milestone 7A Benchmark Suite Generator")
    print("=========================================================")

    # -------------------------------------------------------------------------
    # 1. Genuine Marksheet (Clean Q=85, Valid QR, AARAV SHARMA)
    # -------------------------------------------------------------------------
    print("\n[1/5] Generating genuine_marksheet.jpg (100% Authentic Baseline)...")
    genuine_file = TEST_DOCS_DIR / "genuine_marksheet.jpg"
    canvas_gen = create_scanned_canvas(1000, 1200, seed=42)

    qr_payload_gen = {
        "issuer": "BOARD OF SECONDARY EDUCATION",
        "roll": "BSE-2026-104",
        "name": "AARAV SHARMA",
        "dob": "15-08-2006",
        "grades": {
            "MATH-101": "A",
            "PHY-102": "B",
            "CHEM-103": "A",
            "ENG-104": "E",
            "CS-105": "B"
        },
        "sig": "DSC-9F8A7B"
    }

    subjects_gen = [
        ("MATH-101", "Mathematics", "100", "92", "A"),
        ("PHY-102", "Physics", "100", "84", "B"),
        ("CHEM-103", "Chemistry", "100", "88", "A"),
        ("ENG-104", "English Literature", "100", "42", "E"),
        ("CS-105", "Computer Science", "100", "86", "B")
    ]

    coords_gen = draw_marksheet(
        canvas_gen,
        issuer_title="BOARD OF SECONDARY EDUCATION - OFFICIAL MARKSHEET",
        subtitle="HIGHER SECONDARY EXAMINATION CERTIFICATE & TRANSCRIPT",
        candidate_name="AARAV SHARMA",
        roll_no="BSE-2026-104",
        dob="15-08-2006",
        institution="ST. XAVIER'S SENIOR ACADEMY",
        center_code="CTR-0419 (CENTRAL DISTRICT)",
        date_of_issue="15-JUNE-2024",
        subjects=subjects_gen,
        qr_payload=qr_payload_gen
    )
    canvas_gen.save(str(genuine_file), "JPEG", quality=85)
    print(f"      [OK] Saved genuine_marksheet.jpg ({genuine_file.stat().st_size:,} bytes)")

    # -------------------------------------------------------------------------
    # 2. Tampered Marksheet (Type 1: Edited Grade E -> B, Photoshop EXIF)
    # -------------------------------------------------------------------------
    print("\n[2/5] Generating tampered_marksheet.jpg (Type 1: Edited Genuine Marksheet)...")
    tampered_file = TEST_DOCS_DIR / "tampered_marksheet.jpg"
    img_tamp = Image.open(str(genuine_file)).convert("RGB")
    draw_tamp = ImageDraw.Draw(img_tamp)

    patch_bbox = coords_gen.get("patch_bbox", (830, 562, 900, 602))
    grade_center = coords_gen.get("center", (865, 582))

    # Overwrite Grade 'E' with flat background patch
    draw_tamp.rectangle(patch_bbox, fill=(250, 248, 244))

    # Render Grade 'B' with +3px baseline drift and sharper edges
    font_forged = get_font(15, bold=True)
    shifted_x = grade_center[0]
    shifted_y = grade_center[1] + 3
    draw_tamp.text((shifted_x, shifted_y), "B", fill=(15, 30, 60), font=font_forged, anchor="mm")

    # Add suspicious Software metadata tag (0x0131)
    exif_tamp = img_tamp.getexif()
    exif_tamp[0x0131] = "Adobe Photoshop 2026 (Windows)"
    exif_tamp[0x9286] = "Edited layered file export"

    img_tamp.save(str(tampered_file), "JPEG", quality=95, exif=exif_tamp)
    print(f"      [OK] Saved tampered_marksheet.jpg ({tampered_file.stat().st_size:,} bytes)")
    print("           - Grade 'E' -> 'B' (+3px baseline drift)")
    print("           - EXIF Software: 'Adobe Photoshop 2026'")
    print("           - Embedded QR payload still states 'ENG-104': 'E'")

    # -------------------------------------------------------------------------
    # 3. Fabricated Certificate (Type 2: Fabricated Certificate from Real University)
    # -------------------------------------------------------------------------
    print("\n[3/5] Generating fabricated_certificate.jpg (Type 2: Fabricated Certificate)...")
    fabricated_file = TEST_DOCS_DIR / "fabricated_certificate.jpg"
    canvas_fab = create_scanned_canvas(1000, 1200, seed=101)

    qr_payload_fab = {
        "issuer": "BOARD OF SECONDARY EDUCATION",
        "roll": "BSE-2026-999",  # Does NOT exist in registry
        "name": "ROHAN VERMA",
        "dob": "22-11-2005",
        "grades": {
            "MATH-101": "A",
            "PHY-102": "A",
            "CHEM-103": "A",
            "ENG-104": "A",
            "CS-105": "A"
        },
        "sig": "FAKE-000"  # Invalid signature prefix
    }

    subjects_fab = [
        ("MATH-101", "Mathematics", "100", "98", "A"),
        ("PHY-102", "Physics", "100", "95", "A"),
        ("CHEM-103", "Chemistry", "100", "96", "A"),
        ("ENG-104", "English Literature", "100", "94", "A"),
        ("CS-105", "Computer Science", "100", "99", "A")
    ]

    draw_marksheet(
        canvas_fab,
        issuer_title="BOARD OF SECONDARY EDUCATION - OFFICIAL MARKSHEET",
        subtitle="HIGHER SECONDARY EXAMINATION CERTIFICATE & TRANSCRIPT",
        candidate_name="ROHAN VERMA",
        roll_no="BSE-2026-999",
        dob="22-11-2005",
        institution="DELHI PUBLIC EXCELLENCE SCHOOL",
        center_code="CTR-0812 (NORTH DISTRICT)",
        date_of_issue="18-JUNE-2024",
        subjects=subjects_fab,
        qr_payload=qr_payload_fab
    )
    canvas_fab.save(str(fabricated_file), "JPEG", quality=85)
    print(f"      [OK] Saved fabricated_certificate.jpg ({fabricated_file.stat().st_size:,} bytes)")
    print("           - Roll BSE-2026-999 (Non-existent in official archives)")
    print("           - Fake QR signature: 'FAKE-000'")

    # -------------------------------------------------------------------------
    # 4. Unrecognised Institution (Type 3: Unrecognised Institution)
    # -------------------------------------------------------------------------
    print("\n[4/5] Generating unrecognised_institution.jpg (Type 3: Fake/Unrecognised Board)...")
    unrec_file = TEST_DOCS_DIR / "unrecognised_institution.jpg"
    canvas_unrec = create_scanned_canvas(1000, 1200, seed=202)

    qr_payload_unrec = {
        "issuer": "GLOBAL DIGITAL OPEN UNIVERSITY OF INDIA",
        "roll": "GDOU-2026-555",
        "name": "VIKRAM MEHTA",
        "dob": "10-04-2005",
        "grades": {
            "MATH-101": "A",
            "PHY-102": "B",
            "CHEM-103": "A",
            "ENG-104": "B",
            "CS-105": "B"
        },
        "sig": "DSC-3321AA"
    }

    subjects_unrec = [
        ("MATH-101", "Mathematics", "100", "88", "A"),
        ("PHY-102", "Physics", "100", "82", "B"),
        ("CHEM-103", "Chemistry", "100", "86", "A"),
        ("ENG-104", "English Literature", "100", "80", "B"),
        ("CS-105", "Computer Science", "100", "84", "B")
    ]

    draw_marksheet(
        canvas_unrec,
        issuer_title="GLOBAL DIGITAL OPEN UNIVERSITY OF INDIA",
        subtitle="AUTONOMOUS DISTANCE EDUCATION & ACADEMIC TRANSCRIPT",
        candidate_name="VIKRAM MEHTA",
        roll_no="GDOU-2026-555",
        dob="10-04-2005",
        institution="DIGITAL LEARNING CAMPUS",
        center_code="CTR-9901 (ONLINE)",
        date_of_issue="20-JULY-2024",
        subjects=subjects_unrec,
        qr_payload=qr_payload_unrec
    )
    canvas_unrec.save(str(unrec_file), "JPEG", quality=85)
    print(f"      [OK] Saved unrecognised_institution.jpg ({unrec_file.stat().st_size:,} bytes)")
    print("           - Issuer: 'GLOBAL DIGITAL OPEN UNIVERSITY OF INDIA' (UGC Blacklist)")

    # -------------------------------------------------------------------------
    # 5. Fully AI-Generated Document (Type 4: GenAI Synthetic Document)
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating ai_generated_doc.jpg (Type 4: Generative AI Document)...")
    ai_file = TEST_DOCS_DIR / "ai_generated_doc.jpg"
    canvas_ai = create_scanned_canvas(1000, 1200, seed=333)

    subjects_ai = [
        ("MATH-101", "Mathematics", "100", "95", "A"),
        ("PHY-102", "Physics", "100", "91", "A"),
        ("CHEM-103", "Chemistry", "100", "93", "A"),
        ("ENG-104", "English Literature", "100", "89", "A"),
        ("CS-105", "Computer Science", "100", "96", "A")
    ]

    # No QR code passed (AI models cannot produce valid cryptographic QR codes)
    draw_marksheet(
        canvas_ai,
        issuer_title="NATIONAL AI EXCELLENCE STATEMENT & TRANSCRIPT",
        subtitle="SYNTHETIC PERFORMANCE STATEMENT & HONORS ATTAINMENT",
        candidate_name="AI_SYNTHETIC_CANDIDATE",
        roll_no="GEN-2026-001",
        dob="01-01-2006",
        institution="METAVERSE EDUCATION ACADEMY",
        center_code="CTR-GENAI (VIRTUAL)",
        date_of_issue="01-JANUARY-2026",
        subjects=subjects_ai,
        qr_payload=None  # Missing QR!
    )

    # Set EXIF software tag
    exif_ai = canvas_ai.getexif()
    exif_ai[0x0131] = "Midjourney v6 / Stable Diffusion XL"
    exif_ai[0x9286] = "AI Generated Visual Asset"
    canvas_ai.save(str(ai_file), "JPEG", quality=85, exif=exif_ai)
    print(f"      [OK] Saved ai_generated_doc.jpg ({ai_file.stat().st_size:,} bytes)")
    print("           - Missing QR Code")
    print("           - EXIF Software: 'Midjourney v6 / Stable Diffusion XL'")

    print("\n=========================================================")
    print("  All 5 Benchmark Files Generated in test_docs/          ")
    print("  (Type 5 Tested via genuine_marksheet with Priya Patel) ")
    print("=========================================================")


if __name__ == "__main__":
    generate_benchmark_suite()
