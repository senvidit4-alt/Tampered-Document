"""
DocTamperGuard - Milestone 2: Document Auto-Alignment & Pre-Processing
Detects perspective distortion and background clutter in mobile camera document captures,
applies 4-point perspective transform to normalize to standard 1000x1200 resolution,
and strictly enforces forensic preservation by avoiding re-interpolation on flat digital scans.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


# Default project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DOCS_DIR = PROJECT_ROOT / "test_docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def order_points(pts: Union[np.ndarray, List]) -> np.ndarray:
    """
    Orders 4 corner coordinates as:
    0: Top-Left (TL)
    1: Top-Right (TR)
    2: Bottom-Right (BR)
    3: Bottom-Left (BL)

    Args:
        pts: (4, 2) array-like of (x, y) coordinates.

    Returns:
        np.ndarray of shape (4, 2) in [TL, TR, BR, BL] order with float32 dtype.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum (x + y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference: y - x
    # Top-right has minimum diff (x is large, y is small -> y - x is minimum)
    # Bottom-left has maximum diff (y is large, x is small -> y - x is maximum)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def _is_flat_scan(image: np.ndarray, contour: np.ndarray, area_ratio: float) -> bool:
    """
    Forensic Preservation Heuristic:
    Checks if the detected contour corresponds to an internal border of a document
    that is already a flat digital scan (e.g. ornamental page frame or table margin).
    
    Warping an already-flat scan introduces bilinear/bicubic resampling interpolation,
    which obliterates authentic JPEG DCT 8x8 block artifacts and micro-noise variance.
    """
    # If the contour occupies more than 90% of the canvas, the page itself fills the frame
    if area_ratio >= 0.90:
        return True

    h, w = image.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(contour)

    # If the bounding box of the contour spans almost the entire frame width & height
    if (bw / w) >= 0.92 and (bh / h) >= 0.92:
        return True

    return False


def align_document(
    image_path_or_array: Union[str, Path, np.ndarray],
    target_width: int = 1000,
    target_height: int = 1200,
    min_area_ratio: float = 0.25,
    max_area_ratio: float = 0.95
) -> Dict[str, Union[np.ndarray, Optional[List[List[float]]], bool, Tuple[int, ...]]]:
    """
    Detects document boundary and straightens mobile camera captures.

    Forensic Preservation Rule:
    If the document fills >= 95% of the frame (or is already a flat digital scan),
    the original image is returned UNTOUCHED (was_warped=False) to avoid altering
    delicate forensic ELA compression signatures and local noise variance.

    Args:
        image_path_or_array: Path to image file or existing BGR numpy ndarray.
        target_width: Standardized document width in pixels (default: 1000).
        target_height: Standardized document height in pixels (default: 1200).
        min_area_ratio: Minimum fraction of total image area to qualify as document (0.25).
        max_area_ratio: Maximum fraction before triggering flat-scan preservation (0.95).

    Returns:
        Dict containing:
            - "aligned_image": np.ndarray (BGR image ready for OCR / Forensic analyzers)
            - "corner_points": list of 4 [x, y] coordinates or None
            - "was_warped": bool (True if perspective transformed, False if flat scan)
            - "original_shape": tuple (H, W, C)
    """
    # 1. Load image if path provided
    if isinstance(image_path_or_array, (str, Path)):
        path_str = str(image_path_or_array)
        image = cv2.imread(path_str)
        if image is None:
            # Fallback for paths with special characters on Windows
            file_data = np.fromfile(path_str, dtype=np.uint8)
            image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image from: {path_str}")
    elif isinstance(image_path_or_array, np.ndarray):
        image = image_path_or_array.copy()
    else:
        raise TypeError(f"Expected file path or np.ndarray, got {type(image_path_or_array)}")

    orig_h, orig_w = image.shape[:2]
    total_area = float(orig_h * orig_w)
    original_shape = image.shape

    # 2. Edge & Contour Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dynamic Canny thresholds using image median
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)

    # Dilate slightly to bridge small gaps in document borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_quad = None
    quad_area_ratio = 0.0

    # 3. Find largest 4-point polygon
    for c in contours[:6]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            ratio = area / total_area
            if ratio >= min_area_ratio:
                document_quad = approx.reshape(4, 2)
                quad_area_ratio = ratio
                break

    # 4. Check Forensic Preservation Condition
    # If no 4-point contour is found OR if the document is already a flat digital scan
    if document_quad is None or _is_flat_scan(image, document_quad, quad_area_ratio):
        return {
            "aligned_image": image,
            "corner_points": None,
            "was_warped": False,
            "original_shape": original_shape
        }

    # 5. Apply 4-Point Perspective Transform (Tilted mobile photo detected)
    ordered_corners = order_points(document_quad)

    dst_corners = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
    aligned = cv2.warpPerspective(
        image, M, (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return {
        "aligned_image": aligned,
        "corner_points": ordered_corners.tolist(),
        "was_warped": True,
        "original_shape": original_shape
    }


def draw_alignment_overlay(
    image: np.ndarray,
    corner_points: Optional[Union[List, np.ndarray]]
) -> np.ndarray:
    """
    Renders high-visibility green corner brackets and scan boundary overlay
    inspired by mobile document scanner viewfinders (CamScanner / WhatsApp Doc).

    Args:
        image: Original BGR image.
        corner_points: List or array of 4 ordered corner coordinates [[x, y], ...].

    Returns:
        Annotated BGR image with green scanner overlays.
    """
    overlay = image.copy()
    if corner_points is None or len(corner_points) != 4:
        return overlay

    pts = np.asarray(corner_points, dtype=np.int32).reshape(4, 2)

    # 1. Semi-transparent scan area tint
    tint_layer = overlay.copy()
    cv2.fillPoly(tint_layer, [pts], (0, 255, 0))
    cv2.addWeighted(tint_layer, 0.08, overlay, 0.92, 0, overlay)

    # 2. Outer glow & sharp green perimeter line
    cv2.polylines(overlay, [pts], isClosed=True, color=(20, 50, 20), thickness=4)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 3. Scanner Corner Brackets & Crosshairs
    labels = ["TL", "TR", "BR", "BL"]
    bracket_len = 28

    for idx, (x, y) in enumerate(pts):
        # Base circular reticle
        cv2.circle(overlay, (x, y), 11, (0, 0, 0), -1)
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(overlay, (x, y), 3, (255, 255, 255), -1)

        # L-shaped corner bracket lines
        p_prev = pts[(idx - 1) % 4]
        p_next = pts[(idx + 1) % 4]

        # Unit vectors towards adjacent corners
        v1 = p_prev - np.array([x, y], dtype=np.float32)
        v2 = p_next - np.array([x, y], dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 0:
            dir1 = (v1 / n1) * bracket_len
            pt1 = (int(x + dir1[0]), int(y + dir1[1]))
            cv2.line(overlay, (x, y), pt1, (0, 255, 0), 3)

        if n2 > 0:
            dir2 = (v2 / n2) * bracket_len
            pt2 = (int(x + dir2[0]), int(y + dir2[1]))
            cv2.line(overlay, (x, y), pt2, (0, 255, 0), 3)

        # High-contrast label tag
        tag_x = x + (14 if idx in (0, 3) else -32)
        tag_y = y + (-10 if idx in (0, 1) else 20)
        cv2.putText(
            overlay, labels[idx], (tag_x, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            overlay, labels[idx], (tag_x, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    return overlay


def create_synthetic_tilted_document(
    doc_path: Path,
    canvas_w: int = 1400,
    canvas_h: int = 1600
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a realistic synthetic tilted mobile capture of genuine_marksheet.jpg
    placed on a dark desk surface with perspective angle.
    """
    doc = cv2.imread(str(doc_path))
    if doc is None:
        raise FileNotFoundError(f"Missing base document: {doc_path}")

    dh, dw = doc.shape[:2]

    # Create dark textured wooden desk surface
    bg = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    bg[:] = (32, 28, 26)  # Dark charcoal/wood background

    # Subtle table surface noise
    noise = np.random.normal(0, 3, (canvas_h, canvas_w, 3))
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Define perspective tilt coordinates
    src_corners = np.array([
        [0, 0],
        [dw, 0],
        [dw, dh],
        [0, dh]
    ], dtype=np.float32)

    # Destination coordinates on desk with realistic perspective tilt (~55% area)
    dst_corners = np.array([
        [185.0, 145.0],    # Top-Left
        [1250.0, 115.0],   # Top-Right
        [1305.0, 1485.0],  # Bottom-Right
        [125.0, 1435.0]    # Bottom-Left
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    tilted = cv2.warpPerspective(
        doc, M, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(32, 28, 26)
    )

    return tilted, dst_corners


if __name__ == "__main__":
    print("=========================================================")
    print("  DocTamperGuard - Milestone 2 Alignment Verification    ")
    print("=========================================================")

    tampered_file = TEST_DOCS_DIR / "tampered_marksheet.jpg"
    genuine_file = TEST_DOCS_DIR / "genuine_marksheet.jpg"
    output_aligned_file = OUTPUTS_DIR / "test_aligned.jpg"

    assert tampered_file.exists(), f"Error: {tampered_file} missing!"
    assert genuine_file.exists(), f"Error: {genuine_file} missing!"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # TEST CASE A: Flat Digital Scan (Forensic Preservation Check)
    # -------------------------------------------------------------------------
    print("\n[TEST A] Evaluating tampered_marksheet.jpg (Flat Digital Scan)...")
    res_flat = align_document(tampered_file)

    print(f"  • Original Shape : {res_flat['original_shape']}")
    print(f"  • Aligned Shape  : {res_flat['aligned_image'].shape}")
    print(f"  • was_warped     : {res_flat['was_warped']}")
    print(f"  • corner_points  : {res_flat['corner_points']}")

    # Validation assertions for Test A
    assert res_flat["was_warped"] is False, (
        "FAIL: Flat scan was warped! Forensic preservation rule violated."
    )
    assert res_flat["aligned_image"].shape == res_flat["original_shape"], (
        "FAIL: Image shape was modified!"
    )
    print("  >> PASS: Forensic preservation active. Zero interpolation artifacts introduced.")

    # -------------------------------------------------------------------------
    # TEST CASE B: Synthetic Tilted Mobile Camera Capture
    # -------------------------------------------------------------------------
    print("\n[TEST B] Synthesizing tilted mobile capture of genuine marksheet...")
    tilted_img, ground_truth_corners = create_synthetic_tilted_document(genuine_file)
    print(f"  • Tilted image shape: {tilted_img.shape} on dark surface")

    print("  • Executing auto-alignment & perspective rectification...")
    res_tilted = align_document(tilted_img, target_width=1000, target_height=1200)

    print(f"  • Detected Corners : {res_tilted['corner_points']}")
    print(f"  • was_warped       : {res_tilted['was_warped']}")
    print(f"  • Aligned Shape    : {res_tilted['aligned_image'].shape}")

    # Validation assertions for Test B
    assert res_tilted["was_warped"] is True, "FAIL: Tilted document was not warped!"
    assert res_tilted["aligned_image"].shape == (1200, 1000, 3), (
        f"FAIL: Expected (1200, 1000, 3), got {res_tilted['aligned_image'].shape}"
    )
    assert res_tilted["corner_points"] is not None, "FAIL: Corner points not detected!"
    print("  >> PASS: 4-point perspective warp successfully flattened document to 1000x1200.")

    # -------------------------------------------------------------------------
    # Generate Visual Inspection Composite & Overlays
    # -------------------------------------------------------------------------
    print("\n[EXPORT] Generating visual inspection output to outputs/test_aligned.jpg...")
    overlay_img = draw_alignment_overlay(tilted_img, res_tilted["corner_points"])

    # Build side-by-side verification composite:
    # Left: Mobile Camera Capture with Green Scanner Reticle
    # Right: Rectified & Cropped 1000x1200 Output
    display_h = 900
    display_w_left = int(tilted_img.shape[1] * (display_h / tilted_img.shape[0]))
    left_resized = cv2.resize(overlay_img, (display_w_left, display_h), interpolation=cv2.INTER_AREA)

    aligned_doc = res_tilted["aligned_image"]
    display_w_right = int(aligned_doc.shape[1] * (display_h / aligned_doc.shape[0]))
    right_resized = cv2.resize(aligned_doc, (display_w_right, display_h), interpolation=cv2.INTER_AREA)

    # Combine into horizontal split-view banner
    separator = np.ones((display_h, 16, 3), dtype=np.uint8) * 45
    composite = np.hstack([left_resized, separator, right_resized])

    # Top status banner
    banner_h = 55
    banner = np.zeros((banner_h, composite.shape[1], 3), dtype=np.uint8)
    banner[:] = (20, 24, 32)
    cv2.putText(
        banner,
        "DocTamperGuard [Milestone 2] - LEFT: Detected Mobile Tilt & Corners  |  RIGHT: Rectified (1000x1200)",
        (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, cv2.LINE_AA
    )
    final_output = np.vstack([banner, composite])

    cv2.imwrite(str(output_aligned_file), final_output)
    print(f"  • Verification composite saved to: {output_aligned_file} ({output_aligned_file.stat().st_size:,} bytes)")

    # Also save standalone aligned doc & overlay for downstream modules
    standalone_aligned = OUTPUTS_DIR / "aligned_document.jpg"
    standalone_overlay = OUTPUTS_DIR / "alignment_overlay.jpg"
    cv2.imwrite(str(standalone_aligned), aligned_doc)
    cv2.imwrite(str(standalone_overlay), overlay_img)
    print(f"  • Standalone aligned doc saved to: {standalone_aligned}")
    print(f"  • Standalone overlay saved to    : {standalone_overlay}")

    print("\n=========================================================")
    print("  Milestone 2: PASSED ALL TESTS SUCCESSFULLY!            ")
    print("=========================================================")
