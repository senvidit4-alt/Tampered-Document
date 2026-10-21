"""
DocTamperGuard - Milestone 7A: Master Decision & Policy Engine
Upgraded to the Full "Five-Check Verification Stack" (STEP 0 to CHECK 5)
with Tri-State Verdicts: PASS / REFER / FAIL

Stack Architecture:
  - STEP 0: Document Intake & Semantic Field Extraction
  - CHECK 1: File Forensics & Software Metadata (Photoshop / Generative AI)
  - CHECK 2: Pixel-Level Analysis (ELA Residuals, Sub-Pixel Baseline Drift, Font Sharpness)
  - CHECK 3: Cryptographic QR Code & Self-Consistency Verification
  - CHECK 4: Issuer Recognition (UGC / Board Registry) & DigiLocker Central Archive Lookup
  - CHECK 5: Cross-Document Applicant Identity Match (Anti-Impersonation)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aligner import align_document, draw_alignment_overlay
from core.segmenter import segment_document_targets, draw_segmentation_map
from core.forensics import run_forensic_scan, compute_ela_map
from core.stack_checks import (
    run_step0_intake,
    run_check1_file_forensics,
    run_check3_qr_and_signature,
    run_check4_issuer_source,
    run_check5_cross_document_match
)

TEST_DOCS_DIR = PROJECT_ROOT / "test_docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def render_verdict_overlay(
    aligned_bgr: np.ndarray,
    analyzed_segments: List[Dict],
    flagged_patches: List[Dict],
    status: str,
    tri_state_verdict: str = "PASS",
    failed_checks: Optional[List[str]] = None
) -> np.ndarray:
    """
    Renders high-impact visual inspection overlay:
    - Top banner:
        * Emerald Green for PASS / ACCEPTED
        * Crimson Red for FAIL / REJECTED
        * Amber / Gold for REFER
    - Verified Cells: Crisp emerald green bounding reticles (#00C853).
    - Flagged Patches: Glowing red spotlight halo and target reticle with forensic reasons.
    - Stack Verification Status callout box if non-pixel checks failed.
    """
    canvas = aligned_bgr.copy()
    img_h, img_w = canvas.shape[:2]

    # 1. Draw subtle green verification boxes for authentic grade/mark cells
    for s in analyzed_segments:
        if s.get("is_grade_or_mark", False) and not s.get("is_tampered", False):
            tx1, ty1, tx2, ty2 = s["tight_bbox"]
            cv2.rectangle(canvas, (tx1 - 2, ty1 - 2), (tx2 + 2, ty2 + 2), (83, 200, 0), 1, cv2.LINE_AA)
            for cx, cy in [(tx1 - 2, ty1 - 2), (tx2 + 2, ty1 - 2), (tx1 - 2, ty2 + 2), (tx2 + 2, ty2 + 2)]:
                cv2.circle(canvas, (cx, cy), 1, (83, 200, 0), -1)

    # 2. Draw prominent forensic alerts for flagged patches (Check 2)
    if flagged_patches:
        halo_layer = canvas.copy()
        for p in flagged_patches:
            px1, py1, px2, py2 = p["bbox"]
            cv2.rectangle(halo_layer, (px1 - 8, py1 - 8), (px2 + 8, py2 + 8), (30, 30, 240), -1)
        cv2.addWeighted(halo_layer, 0.22, canvas, 0.78, 0, canvas)

        for p in flagged_patches:
            tx1, ty1, tx2, ty2 = p["tight_bbox"]
            conf = p.get("tamper_confidence", 90.0)
            text_val = p.get("text", "?")

            cv2.rectangle(canvas, (tx1 - 3, ty1 - 3), (tx2 + 3, ty2 + 3), (30, 30, 240), 2, cv2.LINE_AA)

            # L-shaped corner brackets
            blen = 6
            for cx, cy, dx, dy in [
                (tx1 - 3, ty1 - 3, 1, 1),
                (tx2 + 3, ty1 - 3, -1, 1),
                (tx1 - 3, ty2 + 3, 1, -1),
                (tx2 + 3, ty2 + 3, -1, -1)
            ]:
                cv2.line(canvas, (cx, cy), (cx + dx * blen, cy), (0, 0, 255), 3)
                cv2.line(canvas, (cx, cy), (cx, cy + dy * blen), (0, 0, 255), 3)

            # Callout card pointer & badge
            card_x1 = min(img_w - 290, tx2 + 35)
            card_y1 = max(65, ty1 - 25)
            card_w = 260
            card_h = 75
            card_x2 = card_x1 + card_w
            card_y2 = card_y1 + card_h

            cv2.line(canvas, (tx2 + 3, (ty1 + ty2) // 2), (card_x1, card_y1 + card_h // 2), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (tx2 + 3, (ty1 + ty2) // 2), 4, (0, 0, 255), -1)

            cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (0, 0, 255), 2)

            disp_text = text_val if len(text_val) <= 16 else f"{text_val[:14]}..."
            cv2.putText(
                canvas, f"FORGERY: '{disp_text}'",
                (card_x1 + 10, card_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (50, 80, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                canvas, f"Tamper Confidence: {conf:.1f}%",
                (card_x1 + 10, card_y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA
            )
            triggers_str = ", ".join(p.get("triggers_fired", []))[:28]
            cv2.putText(
                canvas, f"Triggers: {triggers_str}",
                (card_x1 + 10, card_y1 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 210, 225), 1, cv2.LINE_AA
            )

    # 3. Stack Alert Badge (when non-pixel checks like QR, Issuer, or Metadata fail)
    if failed_checks and not flagged_patches:
        badge_x1 = 40
        badge_y1 = 80
        badge_w = img_w - 80
        badge_h = min(110, 45 + len(failed_checks) * 22)
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h

        # Semi-transparent dark alert card
        sub_overlay = canvas.copy()
        cv2.rectangle(sub_overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (16, 20, 36), -1)
        cv2.addWeighted(sub_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (30, 30, 240), 2)

        cv2.putText(
            canvas, "SECURITY STACK VIOLATION DETECTED",
            (badge_x1 + 15, badge_y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 80, 255), 2, cv2.LINE_AA
        )
        for i, fc in enumerate(failed_checks[:3]):
            cv2.putText(
                canvas, f"• {fc}",
                (badge_x1 + 15, badge_y1 + 50 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 245), 1, cv2.LINE_AA
            )

    # 4. Master Top Status Header Bar
    banner_h = 54
    banner = np.zeros((banner_h, img_w, 3), dtype=np.uint8)

    if tri_state_verdict == "PASS" or status == "ACCEPTED":
        # Emerald Green Theme
        banner[:] = (18, 48, 22)
        cv2.line(banner, (0, banner_h - 2), (img_w, banner_h - 2), (0, 200, 83), 3)
        cv2.putText(
            banner, "✓ VERDICT: PASS - DOCUMENT AUTHENTIC",
            (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            banner, "5-Check Verification Stack Cleared (100% Pass)",
            (img_w - 430, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 240, 140), 1, cv2.LINE_AA
        )
    elif tri_state_verdict == "REFER" or status == "REFER":
        # Amber / Gold Theme
        banner[:] = (20, 42, 60)
        cv2.line(banner, (0, banner_h - 2), (img_w, banner_h - 2), (0, 185, 255), 3)
        cv2.putText(
            banner, "⏸ VERDICT: REFER - MANUAL REVIEW REQUIRED",
            (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            banner, "Borderline Anomaly / Partial Data Detected",
            (img_w - 410, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA
        )
    else:
        # Crimson Red Theme
        banner[:] = (20, 22, 60)
        cv2.line(banner, (0, banner_h - 2), (img_w, banner_h - 2), (30, 30, 240), 3)
        cv2.putText(
            banner, "⚠ VERDICT: FAIL - DOCUMENT FRAUD DETECTED",
            (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            banner, "Zero-Tolerance Policy Fired (Stack Violation)",
            (img_w - 410, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 120, 255), 1, cv2.LINE_AA
        )

    canvas[0:banner_h, 0:img_w] = banner
    return canvas


def extract_patch_zoom_crops(
    aligned_bgr: np.ndarray,
    ela_heatmap_bgr: np.ndarray,
    flagged_patches: List[Dict],
    zoom_size: int = 180,
    crop_padding: int = 35
) -> List[Dict]:
    """
    Extracts zoomed forensic comparison crops for flagged patches.
    """
    zoom_crops = []
    img_h, img_w = aligned_bgr.shape[:2]

    for idx, p in enumerate(flagged_patches):
        tx1, ty1, tx2, ty2 = p["tight_bbox"]

        cx = (tx1 + tx2) // 2
        cy = (ty1 + ty2) // 2
        half_box = max((tx2 - tx1), (ty2 - ty1)) // 2 + crop_padding

        x1 = max(0, cx - half_box)
        x2 = min(img_w, cx + half_box)
        y1 = max(0, cy - half_box)
        y2 = min(img_h, cy + half_box)

        rgb_crop = aligned_bgr[y1:y2, x1:x2]
        ela_crop = ela_heatmap_bgr[y1:y2, x1:x2]

        if rgb_crop.size == 0 or ela_crop.size == 0:
            continue

        scaled_rgb = cv2.resize(rgb_crop, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
        scaled_ela = cv2.resize(ela_crop, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)

        cv2.rectangle(scaled_rgb, (4, 4), (zoom_size - 4, zoom_size - 4), (0, 0, 255), 2)
        cv2.rectangle(scaled_ela, (4, 4), (zoom_size - 4, zoom_size - 4), (0, 185, 255), 2)

        zoom_crops.append({
            "patch_bgr": scaled_rgb,
            "ela_crop_bgr": scaled_ela,
            "label": f"Target #{idx + 1}: '{p['text']}' (Conf: {p.get('tamper_confidence', 0):.1f}%)",
            "text": p["text"],
            "confidence": p.get("tamper_confidence", 0.0),
            "reasons": p.get("forensic_reasons", [])
        })

    return zoom_crops


def render_pdf_to_bgr(pdf_source: Union[bytes, str, Path], page_idx: int = 0, scale: float = 2.0) -> np.ndarray:
    """
    Renders a page of a PDF document into a high-resolution lossless BGR numpy array.
    Uses pypdfium2 (built on Google PDFium engine) requiring zero external binaries.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError("pypdfium2 is required for PDF rendering. Install it via `pip install pypdfium2`.")

    pdf = pdfium.PdfDocument(pdf_source)
    if len(pdf) == 0:
        raise ValueError("Provided PDF document contains no pages.")

    page = pdf[page_idx]
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    arr = np.array(pil_image)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def verify_document(
    image_input: Union[str, Path, bytes, np.ndarray],
    applicant_name: Optional[str] = None,
    applicant_dob: Optional[str] = None,
    save_artifacts_dir: Optional[Union[str, Path]] = None
) -> Dict:
    """
    Master unified entry point for DocTamperGuard:
    Executes the full Five-Check Verification Stack (STEP 0 to CHECK 5)
    returning tri-state verdicts: PASS / REFER / FAIL.
    Supports JPG, PNG, TIFF, and PDF multi-format ingestion.

    Args:
        image_input: File path (str/Path), raw image/PDF bytes, or loaded BGR numpy array.
        applicant_name: Optional applicant name from application form (for Check 5).
        applicant_dob: Optional applicant DOB from application form (for Check 5).
        save_artifacts_dir: Optional directory to persist visual outputs and JSON audit.

    Returns:
        Structured master verification dictionary including verdict, confidence,
        five_check_stack, audit data, and all visual inspection artifacts.
    """
    t_start = time.perf_counter()

    # 1. Decode bytes or PDF if raw file stream passed
    raw_bytes = None
    if isinstance(image_input, bytes):
        raw_bytes = image_input
        if raw_bytes.startswith(b"%PDF"):
            image_data = render_pdf_to_bgr(raw_bytes)
        else:
            file_arr = np.frombuffer(image_input, dtype=np.uint8)
            loaded_bgr = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
            if loaded_bgr is None:
                raise ValueError("Failed to decode image from provided byte stream.")
            image_data = loaded_bgr
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.exists():
            raw_bytes = p.read_bytes()
            if p.suffix.lower() == ".pdf" or raw_bytes.startswith(b"%PDF"):
                image_data = render_pdf_to_bgr(str(p))
            else:
                loaded_bgr = cv2.imread(str(p))
                if loaded_bgr is None:
                    raise ValueError(f"Failed to read image from path: {p}")
                image_data = loaded_bgr
        else:
            raise FileNotFoundError(f"Input image not found: {image_input}")
    else:
        image_data = image_input

    # -------------------------------------------------------------------------
    # STAGE 1: Document Alignment & Pre-Processing
    # -------------------------------------------------------------------------
    align_result = align_document(image_data)
    aligned_bgr = align_result["aligned_image"]
    was_warped = align_result["was_warped"]
    corner_points = align_result["corner_points"]

    if was_warped and corner_points is not None:
        alignment_overlay_bgr = draw_alignment_overlay(
            align_result.get("original_image", image_data),
            corner_points
        )
    else:
        alignment_overlay_bgr = aligned_bgr.copy()

    # -------------------------------------------------------------------------
    # STAGE 2: Deep Target Segmentation & Row-Grid Mapping
    # -------------------------------------------------------------------------
    segments = segment_document_targets(aligned_bgr)
    segmentation_map_bgr = draw_segmentation_map(aligned_bgr, segments)

    # -------------------------------------------------------------------------
    # STEP 0: Intelligent Document Intake & Semantic Entity Extraction
    # -------------------------------------------------------------------------
    step0_data = run_step0_intake(segments, image_input=aligned_bgr)

    # -------------------------------------------------------------------------
    # CHECK 1: File Forensics & Metadata Scanner (Photoshop / GenAI)
    # -------------------------------------------------------------------------
    check1_input = raw_bytes if raw_bytes is not None else image_input
    check1_res = run_check1_file_forensics(check1_input)
    is_manipulated_or_genai = (check1_res.get("status") == "FAIL")

    # -------------------------------------------------------------------------
    # CHECK 3: Cryptographic QR Code & Self-Consistency
    # -------------------------------------------------------------------------
    check3_res = run_check3_qr_and_signature(
        aligned_bgr,
        step0_data,
        is_synthetic_or_manipulated=is_manipulated_or_genai
    )
    is_digilocker = bool(check3_res.get("digilocker_verified", False))
    if is_digilocker and is_manipulated_or_genai:
        check1_res = run_check1_file_forensics(check1_input, digilocker_verified=True)

    # -------------------------------------------------------------------------
    # CHECK 2: Pixel-Level Analysis (ELA Residuals, Baseline Drift, Sharpness)
    # -------------------------------------------------------------------------
    forensic_result = run_forensic_scan(aligned_bgr, segments, digilocker_verified=is_digilocker)
    ela_heatmap_bgr = forensic_result["ela_heatmap"]
    analyzed_segments = forensic_result["analyzed_segments"]
    flagged_patches = forensic_result["flagged_patches"]
    max_pixel_confidence = forensic_result["max_tamper_confidence"]

    # Cross-signal container forensics: Correlate stripped EXIF with Check 2 pixel micro-forgeries
    if check1_res.get("is_stripped_exif") and len(flagged_patches) > 0 and check1_res.get("status") == "PASS" and not is_digilocker:
        check1_res = run_check1_file_forensics(check1_input, pixel_flagged_count=len(flagged_patches))
        is_manipulated_or_genai = (check1_res.get("status") == "FAIL")

    check2_status = "FAIL" if len(flagged_patches) > 0 else "PASS"
    check2_title = (
        f"Pixel Micro-Forgery Detected ({len(flagged_patches)} cell(s))"
        if len(flagged_patches) > 0
        else "Authentic Pixel & Compression Profile"
    )
    check2_details = (
        [f"{len(flagged_patches)} tampered micro-patch(es) isolated."] +
        [r for p in flagged_patches for r in p.get("forensic_reasons", [])]
        if len(flagged_patches) > 0
        else [
            "No spatial baseline drift, ELA anomalies, or font sharpness mismatches.",
            "All grade cells adhere to physical scanner statistical profile."
        ]
    )

    check2_res = {
        "check_id": "CHECK_2",
        "check_name": "Check 2: Pixel Analysis (ELA, Drift & Sharpness)",
        "status": check2_status,
        "verdict_title": check2_title,
        "flagged_count": len(flagged_patches),
        "tamper_confidence": max_pixel_confidence,
        "details": check2_details
    }

    # -------------------------------------------------------------------------
    # CHECK 4: Issuer Recognition & Central Archive Database Lookup
    # -------------------------------------------------------------------------
    check4_res = run_check4_issuer_source(step0_data, check3_res)

    # -------------------------------------------------------------------------
    # CHECK 5: Cross-Document Applicant Identity Match
    # -------------------------------------------------------------------------
    check5_res = run_check5_cross_document_match(
        step0_data,
        applicant_name=applicant_name,
        applicant_dob=applicant_dob
    )

    # -------------------------------------------------------------------------
    # FIVE-CHECK STACK AGGREGATION & TRI-STATE VERDICT
    # -------------------------------------------------------------------------
    five_check_stack = {
        "step0": {
            "check_id": "STEP_0",
            "check_name": "Step 0: Intake & Semantic Extraction",
            "status": "PASS" if step0_data.get("candidate_name") != "UNKNOWN" else "REFER",
            "verdict_title": "Semantic Document Entity Intake",
            "data": step0_data,
            "details": [
                f"Candidate: {step0_data.get('candidate_name')}",
                f"Roll Number: {step0_data.get('roll_no')}",
                f"Institution: {step0_data.get('institution_name')}",
                f"Extracted Grades: {step0_data.get('extracted_grades')}"
            ]
        },
        "check1": check1_res,
        "check2": check2_res,
        "check3": check3_res,
        "check4": check4_res,
        "check5": check5_res
    }

    # Evaluate checks for overall verdict
    active_checks = [check1_res, check2_res, check3_res, check4_res, check5_res]
    failed_check_titles = [c["verdict_title"] for c in active_checks if c["status"] == "FAIL"]
    refer_check_titles = [c["verdict_title"] for c in active_checks if c["status"] == "REFER"]

    if failed_check_titles:
        tri_state_verdict = "FAIL"
        status = "REJECTED"
        is_authentic = False
        confidence_score = max(max_pixel_confidence, 94.0)
        summary_msg = (
            f"VERIFICATION FAILED: {len(failed_check_titles)} check(s) flagged fraud: "
            + " | ".join(failed_check_titles)
        )
    elif refer_check_titles:
        tri_state_verdict = "REFER"
        status = "REFER"
        is_authentic = False
        confidence_score = 65.0
        summary_msg = (
            f"VERIFICATION HELD (REFER): Manual inspection advised. Warnings: "
            + " | ".join(refer_check_titles)
        )
    else:
        tri_state_verdict = "PASS"
        status = "ACCEPTED"
        is_authentic = True
        confidence_score = 99.4
        summary_msg = (
            "ALL 5 CHECKS PASSED: Document verified 100% authentic across File Forensics, "
            "Pixel Analysis, Cryptographic QR, Central Archive Registry, and Applicant Matching."
        )

    # 5. Visual Artifact Synthesis
    verdict_overlay_bgr = render_verdict_overlay(
        aligned_bgr,
        analyzed_segments,
        flagged_patches,
        status=status,
        tri_state_verdict=tri_state_verdict,
        failed_checks=failed_check_titles
    )
    zoom_crops = extract_patch_zoom_crops(
        aligned_bgr, ela_heatmap_bgr, flagged_patches, zoom_size=180
    )

    t_end = time.perf_counter()
    processing_time_ms = round((t_end - t_start) * 1000.0, 1)

    # 6. Structured JSON-Serializable Audit Data
    counts = {
        "total_segments": len(segments),
        "grade_mark_cells": sum(1 for s in segments if s.get("is_grade_or_mark", False)),
        "flagged_patches": len(flagged_patches),
        "checks_passed": sum(1 for c in active_checks if c["status"] == "PASS"),
        "checks_failed": len(failed_check_titles),
        "checks_refer": len(refer_check_titles)
    }

    flagged_details = []
    for p in flagged_patches:
        flagged_details.append({
            "id": p.get("id"),
            "text": p.get("text"),
            "tight_bbox": p.get("tight_bbox"),
            "padded_bbox": p.get("bbox"),
            "row_id": p.get("row_id"),
            "tamper_confidence": p.get("tamper_confidence"),
            "triggers_fired": p.get("triggers_fired", []),
            "forensic_reasons": p.get("forensic_reasons", []),
            "baseline_drift_px": p.get("baseline_drift_px", 0.0),
            "ela_p90": p.get("ela_p90", 0.0)
        })

    audit_report = {
        "engine": "DocTamperGuard 5-Check Verification Engine v2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": tri_state_verdict,
        "status": status,
        "is_authentic": is_authentic,
        "confidence_score": confidence_score,
        "processing_time_ms": processing_time_ms,
        "summary": summary_msg,
        "applicant_input": {
            "applicant_name": applicant_name,
            "applicant_dob": applicant_dob
        },
        "five_check_stack": five_check_stack,
        "alignment": {
            "was_warped": was_warped,
            "corners_detected": corner_points is not None
        },
        "counts": counts,
        "flagged_violations": flagged_details
    }

    # Optional: Save artifacts to disk
    if save_artifacts_dir:
        out_dir = Path(save_artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "verdict_overlay.jpg"), verdict_overlay_bgr)
        cv2.imwrite(str(out_dir / "ela_heatmap.jpg"), ela_heatmap_bgr)
        cv2.imwrite(str(out_dir / "segmentation_map.jpg"), segmentation_map_bgr)
        with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

    return {
        "verdict": tri_state_verdict,
        "status": status,
        "is_authentic": is_authentic,
        "confidence_score": confidence_score,
        "processing_time_ms": processing_time_ms,
        "summary_message": summary_msg,
        "five_check_stack": five_check_stack,
        "alignment_info": {
            "was_warped": was_warped,
            "corner_points": corner_points
        },
        "counts": counts,
        "flagged_details": flagged_details,
        "visuals": {
            "aligned_bgr": aligned_bgr,
            "alignment_overlay_bgr": alignment_overlay_bgr,
            "segmentation_map_bgr": segmentation_map_bgr,
            "ela_heatmap_bgr": ela_heatmap_bgr,
            "verdict_overlay_bgr": verdict_overlay_bgr,
            "zoom_crops": zoom_crops
        },
        "audit_report": audit_report
    }


if __name__ == "__main__":
    print("=========================================================")
    print("  DocTamperGuard - Milestone 7A 5-Check Stack Evaluator   ")
    print("=========================================================")

    test_cases = [
        {
            "name": "Scenario 0: Authentic Baseline Scan",
            "file": "genuine_marksheet.jpg",
            "applicant_name": "AARAV SHARMA",
            "expected_verdict": "PASS"
        },
        {
            "name": "Scenario 1: Edited Genuine Marksheet",
            "file": "tampered_marksheet.jpg",
            "applicant_name": "AARAV SHARMA",
            "expected_verdict": "FAIL"
        },
        {
            "name": "Scenario 2: Fabricated Certificate (Real University)",
            "file": "fabricated_certificate.jpg",
            "applicant_name": "ROHAN VERMA",
            "expected_verdict": "FAIL"
        },
        {
            "name": "Scenario 3: Unrecognised Institution / Diploma Mill",
            "file": "unrecognised_institution.jpg",
            "applicant_name": "KAVYA JOSHI",
            "expected_verdict": "FAIL"
        },
        {
            "name": "Scenario 4: Fully AI-Generated Document",
            "file": "ai_generated_doc.jpg",
            "applicant_name": "VIKRAM SINGHANIA",
            "expected_verdict": "FAIL"
        },
        {
            "name": "Scenario 5: Genuine Document, Wrong Person (Impersonation)",
            "file": "genuine_marksheet.jpg",
            "applicant_name": "PRIYA PATEL",  # Genuine Aarav Sharma doc submitted by Priya Patel
            "expected_verdict": "FAIL"
        }
    ]

    results_table = []
    for tc in test_cases:
        fpath = TEST_DOCS_DIR / tc["file"]
        print(f"\nEvaluating: {tc['name']} ({tc['file']})...")
        res = verify_document(fpath, applicant_name=tc["applicant_name"])
        stack = res["five_check_stack"]
        verdict = res["verdict"]

        c1 = stack["check1"]["status"]
        c2 = stack["check2"]["status"]
        c3 = stack["check3"]["status"]
        c4 = stack["check4"]["status"]
        c5 = stack["check5"]["status"]

        passed_test = (verdict == tc["expected_verdict"])
        print(f"  Result: Verdict={verdict} (Expected={tc['expected_verdict']}) | C1={c1} C2={c2} C3={c3} C4={c4} C5={c5}")
        print(f"  Summary: {res['summary_message']}")

        results_table.append({
            "scenario": tc["name"],
            "verdict": verdict,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
            "ok": "PASS" if passed_test else "FAIL"
        })

    print("\n" + "=" * 105)
    print("  MILESTONE 7A: FIVE-CHECK VERIFICATION STACK BENCHMARK MATRIX")
    print("=" * 105)
    print(f"{'Fraud Scenario':<45} | {'Verdict':<8} | {'C1':<5} {'C2':<5} {'C3':<5} {'C4':<5} {'C5':<5} | {'Test'}")
    print("-" * 105)
    for r in results_table:
        print(f"{r['scenario']:<45} | {r['verdict']:<8} | {r['c1']:<5} {r['c2']:<5} {r['c3']:<5} {r['c4']:<5} {r['c5']:<5} | {r['ok']}")
    print("=" * 105)
