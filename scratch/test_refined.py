import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from core.segmenter import segment_document_targets
from core.forensics import compute_ela_map, _measure_boundary_contrast, _compute_background_noise_std

def test_image(img_path: str):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Cannot load", img_path)
        return
        
    img_h, img_w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    raw_ela, _ = compute_ela_map(img_bgr)
    segs = segment_document_targets(img_bgr)
    
    neighborhood_pad = 12
    enriched = []
    for s in segs:
        d = dict(s)
        tx1, ty1, tx2, ty2 = d["tight_bbox"]
        
        # Exclude bottom signature/stamp area (y > 900)
        cy = (ty1 + ty2) / 2.0
        if cy > 900:
            d["is_tampered"] = False
            d["triggers_fired"] = []
            d["forensic_reasons"] = []
            d["tamper_confidence"] = 0.0
            enriched.append(d)
            continue
            
        nx1 = max(0, tx1 - neighborhood_pad)
        ny1 = max(0, ty1 - neighborhood_pad)
        nx2 = min(img_w, tx2 + neighborhood_pad)
        ny2 = min(img_h, ty2 + neighborhood_pad)
        
        ela_crop = raw_ela[ny1:ny2, nx1:nx2]
        gray_crop = img_gray[ny1:ny2, nx1:nx2]
        
        bg_noise_std, bg_mean, flat_bg_ratio = _compute_background_noise_std(gray_crop)
        boundary_step = _measure_boundary_contrast(img_bgr, d["tight_bbox"])
        
        char_crop = img_gray[max(0, ty1):min(img_h, ty2), max(0, tx1):min(img_w, tx2)]
        lap_var = float(cv2.Laplacian(char_crop, cv2.CV_32F).var()) if char_crop.size >= 16 else 0.0
        
        ink_mask = char_crop < 185
        if np.sum(ink_mask) >= 6:
            p10_ink_luma = float(np.percentile(char_crop[ink_mask], 10))
        else:
            p10_ink_luma = 128.0
            
        if char_crop.size >= 16:
            cgx = cv2.Sobel(char_crop, cv2.CV_32F, 1, 0, ksize=3)
            cgy = cv2.Sobel(char_crop, cv2.CV_32F, 0, 1, ksize=3)
            cgrad = np.sqrt(cgx**2 + cgy**2)
            edge_gradient_p90 = float(np.percentile(cgrad, 90))
        else:
            edge_gradient_p90 = 0.0
            
        d["ela_p90"] = float(np.percentile(ela_crop, 90)) if ela_crop.size > 0 else 0.0
        d["bg_noise_std"] = round(bg_noise_std, 3)
        d["bg_mean"] = round(bg_mean, 2)
        d["flat_bg_ratio"] = round(flat_bg_ratio, 3)
        d["boundary_step"] = round(boundary_step, 2)
        d["sharpness"] = round(lap_var, 1)
        d["p10_ink_luma"] = round(p10_ink_luma, 1)
        d["edge_gradient_p90"] = round(edge_gradient_p90, 1)
        enriched.append(d)
        
    candidates = [s for s in enriched if s.get("cy", (s["tight_bbox"][1]+s["tight_bbox"][3])/2.0) <= 900]
    peer_noises = [s["bg_noise_std"] for s in candidates if s["bg_noise_std"] > 0]
    peer_inks = [s["p10_ink_luma"] for s in candidates if s["p10_ink_luma"] < 100]
    peer_edges = [s["edge_gradient_p90"] for s in candidates if s["edge_gradient_p90"] > 0]
    peer_boundaries = [s["boundary_step"] for s in candidates if s["boundary_step"] > 0]
    peer_ela_p90s = [s["ela_p90"] for s in candidates]
    peer_bg_means = [s["bg_mean"] for s in candidates]
    
    median_noise = float(np.median(peer_noises)) if peer_noises else 3.5
    median_ink = float(np.median(peer_inks)) if peer_inks else 50.0
    median_edge = float(np.median(peer_edges)) if peer_edges else 500.0
    median_boundary = float(np.median(peer_boundaries)) if peer_boundaries else 2.0
    median_bg_mean = float(np.median(peer_bg_means)) if peer_bg_means else 245.0
    median_ela_p90 = float(np.median(peer_ela_p90s)) if peer_ela_p90s else 1.0
    
    flagged = []
    for s in candidates:
        txt = s.get("text", "").strip()
        if not txt:
            continue
            
        drift = s["baseline_drift_px"]
        drift_abs = abs(drift)
        ela_p90 = s["ela_p90"]
        boundary_step = s["boundary_step"]
        noise_val = s["bg_noise_std"]
        sharp_val = s["sharpness"]
        bg_val = s["bg_mean"]
        ink_val = s["p10_ink_luma"]
        edge_grad = s["edge_gradient_p90"]
        is_grade = s.get("is_grade_or_mark", False)
        
        triggers = []
        
        # A. Baseline drift
        if is_grade and drift_abs >= 1.8:
            triggers.append("BASELINE_DRIFT")
            
        # B. Digital whiteout (must be bright whiteout bg >= 251.0 higher than paper)
        if bg_val >= 251.0 and (bg_val - median_bg_mean) >= 5.0:
            triggers.append("DIGITAL_WHITEOUT_PATCH")
        elif boundary_step >= 6.0 and boundary_step >= 2.0 * (median_boundary + 1.0):
            triggers.append("DIGITAL_WHITEOUT_PATCH")
        elif median_noise >= 3.0 and noise_val <= 0.35 * median_noise and s["flat_bg_ratio"] >= 0.60:
            triggers.append("DIGITAL_WHITEOUT_PATCH")
            
        # C. ELA Discontinuity
        ela_ratio = (ela_p90 + 1e-4) / (median_ela_p90 + 1e-4)
        if (ela_ratio >= 3.0 and ela_p90 >= 4.0) or (ela_ratio >= 2.2 and ela_p90 >= 6.0) or ela_p90 >= 12.0:
            triggers.append("ELA_DISCONTINUITY")
            
        # D. Font Ink Mismatch (pure digital black ink)
        if median_ink >= 25.0 and ink_val <= 10.0:
            triggers.append("FONT_INK_MISMATCH")
        elif median_edge >= 25.0 and edge_grad >= 2.5 * median_edge:
            triggers.append("FONT_INK_MISMATCH")
            
        is_tampered = False
        if len(triggers) >= 2:
            is_tampered = True
        elif len(triggers) == 1:
            if "DIGITAL_WHITEOUT_PATCH" in triggers and (bg_val >= 252.0 and (bg_val - median_bg_mean) >= 5.5):
                is_tampered = True
            elif "DIGITAL_WHITEOUT_PATCH" in triggers and (median_noise >= 3.0 and noise_val <= 0.32 * median_noise):
                is_tampered = True
            elif "FONT_INK_MISMATCH" in triggers and (median_ink >= 25.0 and ink_val <= 8.0):
                is_tampered = True
            elif "BASELINE_DRIFT" in triggers and drift_abs >= 2.5:
                is_tampered = True
            elif "ELA_DISCONTINUITY" in triggers and (ela_ratio >= 3.0 and ela_p90 >= 8.0):
                is_tampered = True
                
        if is_tampered:
            flagged.append((txt, s["tight_bbox"], triggers))
            
    print(f"\n[{img_path}] Flagged count: {len(flagged)}")
    for f in flagged:
        print(" ->", f)

if __name__ == "__main__":
    for p in ["test_docs/genuine_marksheet.jpg", "test_docs/tampered_marksheet.jpg", "outputs/simulated_sejda_tampered.jpg"]:
        test_image(p)
