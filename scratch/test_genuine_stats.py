import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from core.segmenter import segment_document_targets
from core.forensics import compute_ela_map

def test_forensics_on_genuine():
    img_bgr = cv2.imread("test_docs/genuine_marksheet.jpg")
    img_h, img_w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    raw_ela, _ = compute_ela_map(img_bgr)
    segs = segment_document_targets(img_bgr)
    
    neighborhood_pad = 12
    enriched = []
    for s in segs:
        d = dict(s)
        tx1, ty1, tx2, ty2 = d["tight_bbox"]
        nx1 = max(0, tx1 - neighborhood_pad)
        ny1 = max(0, ty1 - neighborhood_pad)
        nx2 = min(img_w, tx2 + neighborhood_pad)
        ny2 = min(img_h, ty2 + neighborhood_pad)
        
        ela_crop = raw_ela[ny1:ny2, nx1:nx2]
        gray_crop = img_gray[ny1:ny2, nx1:nx2]
        
        # bg noise and flat bg ratio
        bg_mask = gray_crop >= 190
        blurred = cv2.GaussianBlur(gray_crop, (3, 3), 0)
        high_freq = gray_crop.astype(np.float32) - blurred.astype(np.float32)
        sobelx = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        if np.sum(bg_mask) >= 15:
            bg_noise = float(np.std(high_freq[bg_mask]))
            bg_mean = float(np.mean(gray_crop[bg_mask]))
            flat_bg_ratio = float(np.mean(grad_mag[bg_mask] < 1.5))
        else:
            bg_noise = float(np.std(high_freq)) if high_freq.size > 0 else 0.0
            bg_mean = float(np.mean(gray_crop)) if gray_crop.size > 0 else 240.0
            flat_bg_ratio = float(np.mean(grad_mag < 1.5)) if grad_mag.size > 0 else 0.0
            
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
            edge_grad_p90 = float(np.percentile(cgrad, 90))
        else:
            edge_grad_p90 = 0.0
            
        d["ela_p90"] = float(np.percentile(ela_crop, 90)) if ela_crop.size > 0 else 0.0
        d["bg_noise_std"] = bg_noise
        d["bg_mean"] = bg_mean
        d["flat_bg_ratio"] = flat_bg_ratio
        d["sharpness"] = lap_var
        d["p10_ink_luma"] = p10_ink_luma
        d["edge_grad_p90"] = edge_grad_p90
        d["boundary_step"] = 0.0  # simplified for test
        enriched.append(d)
        
    peer_noises = [x["bg_noise_std"] for x in enriched if x["bg_noise_std"] > 0]
    peer_inks = [x["p10_ink_luma"] for x in enriched if x["p10_ink_luma"] < 100]
    peer_edges = [x["edge_grad_p90"] for x in enriched if x["edge_grad_p90"] > 0]
    peer_flats = [x["flat_bg_ratio"] for x in enriched]
    
    median_noise = float(np.median(peer_noises))
    median_ink = float(np.median(peer_inks))
    median_edge = float(np.median(peer_edges))
    median_flat = float(np.median(peer_flats))
    
    print(f"Stats: noise={median_noise:.1f}, ink={median_ink:.1f}, edge={median_edge:.1f}, flat={median_flat:.2f}")
    
    # Check if any segment in genuine_marksheet triggers false alarm
    false_alarms = []
    for s in enriched:
        trigs = []
        if median_noise >= 2.5 and s["bg_noise_std"] <= 0.35 * median_noise:
            trigs.append("NOISE_LOW")
        if s["flat_bg_ratio"] >= 0.70 and median_flat <= 0.35:
            trigs.append("FLAT_BG")
        if median_ink >= 22.0 and s["p10_ink_luma"] <= 10.0:
            trigs.append("INK_DARK")
        if median_edge >= 25.0 and s["edge_grad_p90"] >= 2.2 * median_edge:
            trigs.append("EDGE_SPIKE")
        if trigs:
            false_alarms.append((s["text"], trigs))
            
    print(f"False alarms on genuine_marksheet: {len(false_alarms)}")
    if false_alarms:
        print("Details:", false_alarms)

if __name__ == "__main__":
    test_forensics_on_genuine()
