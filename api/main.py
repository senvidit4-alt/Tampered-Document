"""
DocTamperGuard - Milestone 7A: FastAPI Backend Engine
Serves high-performance REST API endpoints for document forgery detection,
orchestrating the Five-Check Verification Stack (STEP 0 to CHECK 5)
with Tri-State Verdicts (PASS / REFER / FAIL) and base64 visual inspection payloads.
"""

import base64
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.verifier import verify_document

# Directory setup
TEST_DOCS_DIR = PROJECT_ROOT / "test_docs"

app = FastAPI(
    title="DocTamperGuard API",
    description="Five-Check Document Forgery & Identity Tampering Verification Engine",
    version="2.0.0"
)

# Enable CORS for local development & modern frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def bgr_to_base64(img_bgr: Optional[np.ndarray], quality: int = 88) -> str:
    """
    Converts a BGR numpy array image to an inline base64 JPEG data URI string.
    """
    if img_bgr is None or img_bgr.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/api/health")
def health_check():
    """
    Service health check endpoint with stack capabilities.
    """
    return {
        "status": "online",
        "engine": "DocTamperGuard 5-Check Verification API v2.0",
        "verdict_types": ["PASS", "REFER", "FAIL"],
        "stack_checks": [
            "Step 0: Intake & Semantic Entity Extraction",
            "Check 1: File Forensics & Software Metadata (Photoshop/GenAI/Sejda/PDF)",
            "Check 2: Pixel-Level Analysis (ELA Residuals, Baseline Drift, Sharpness, Whiteout)",
            "Check 3: Cryptographic QR Code & Self-Consistency",
            "Check 4: Issuer Recognition (UGC/Board) & Central Archive Lookup",
            "Check 5: Cross-Document Applicant Identity Match"
        ]
    }


@app.get("/api/samples")
def list_samples():
    """
    Returns benchmark test documents and fraud scenario presets in test_docs/.
    """
    sample_presets = [
        {
            "id": "scenario_0",
            "filename": "genuine_marksheet.jpg",
            "label": "Genuine Baseline Scan (100% Authentic)",
            "fraud_type": "None (Authentic Baseline)",
            "expected_verdict": "PASS",
            "recommended_applicant": "AARAV SHARMA",
            "description": "Standard untouched scan, valid DSC root signature, registered with central board archives."
        },
        {
            "id": "scenario_1",
            "filename": "tampered_marksheet.jpg",
            "label": "Type 1: Edited Genuine Marksheet",
            "fraud_type": "Type 1: Pixel Tamper + Photoshop EXIF + QR Contradiction",
            "expected_verdict": "FAIL",
            "recommended_applicant": "AARAV SHARMA",
            "description": "ENG-104 grade modified from 'E' to 'B' (+3px spatial drift, Photoshop metadata, contradicts signed QR)."
        },
        {
            "id": "scenario_2",
            "filename": "fabricated_certificate.jpg",
            "label": "Type 2: Fabricated Certificate (Real University)",
            "fraud_type": "Type 2: Forged QR Signature + Non-Existent Archive Roll",
            "expected_verdict": "FAIL",
            "recommended_applicant": "ROHAN VERMA",
            "description": "Clean Q=85 certificate with real issuer header, but forged 'FAKE-000' QR seal and non-existent roll BSE-2026-999."
        },
        {
            "id": "scenario_3",
            "filename": "unrecognised_institution.jpg",
            "label": "Type 3: Unrecognised Institution / Diploma Mill",
            "fraud_type": "Type 3: Blacklisted / Unaccredited Board",
            "expected_verdict": "FAIL",
            "recommended_applicant": "KAVYA JOSHI",
            "description": "Issued by 'GLOBAL DIGITAL OPEN UNIVERSITY OF INDIA', an unaccredited institution on the UGC blacklist."
        },
        {
            "id": "scenario_4",
            "filename": "ai_generated_doc.jpg",
            "label": "Type 4: Fully AI-Generated Document",
            "fraud_type": "Type 4: Midjourney/SDXL Metadata + Missing Cryptographic QR",
            "expected_verdict": "FAIL",
            "recommended_applicant": "VIKRAM SINGHANIA",
            "description": "Digitally generated certificate with GenAI metadata markers and missing official 2D verification seal."
        },
        {
            "id": "scenario_5",
            "filename": "genuine_marksheet.jpg",
            "label": "Type 5: Genuine Document, Wrong Person (Impersonation)",
            "fraud_type": "Type 5: Cross-Document Applicant Identity Mismatch",
            "expected_verdict": "FAIL",
            "recommended_applicant": "PRIYA PATEL",
            "description": "100% genuine marksheet belonging to Aarav Sharma, but submitted by applicant Priya Patel."
        }
    ]

    # Verify file sizes
    for s in sample_presets:
        p = TEST_DOCS_DIR / s["filename"]
        s["size_bytes"] = p.stat().st_size if p.exists() else 0

    return {"samples": sample_presets}


@app.post("/api/verify-sample/{filename}")
def verify_sample(
    filename: str,
    applicant_name: Optional[str] = Query(None, description="Optional applicant name for Check 5 cross-document verification"),
    applicant_dob: Optional[str] = Query(None, description="Optional applicant DOB for Check 5")
):
    """
    Directly verifies a pre-generated benchmark sample file from test_docs/.
    """
    file_path = TEST_DOCS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found in test_docs.")

    return _process_and_serialize(file_path, applicant_name=applicant_name, applicant_dob=applicant_dob)


@app.post("/api/verify")
async def verify_upload(
    file: UploadFile = File(...),
    applicant_name: str = Form("", description="Applicant name entered in application form"),
    applicant_dob: str = Form("", description="Applicant date of birth")
):
    """
    Receives an uploaded document file and optional applicant form data,
    runs the full Five-Check Verification Stack (STEP 0 to CHECK 5),
    and returns comprehensive JSON audit data with base64 visual inspection maps.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename received.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    clean_name = applicant_name.strip() if applicant_name else None
    clean_dob = applicant_dob.strip() if applicant_dob else None

    return _process_and_serialize(contents, applicant_name=clean_name, applicant_dob=clean_dob)


def _process_and_serialize(
    image_input: Union[bytes, Path],
    applicant_name: Optional[str] = None,
    applicant_dob: Optional[str] = None
) -> Dict:
    """
    Executes verify_document across the 5-check stack and serializes all numpy visuals
    into base64 data URIs for client-side rendering.
    """
    try:
        raw_result = verify_document(
            image_input,
            applicant_name=applicant_name,
            applicant_dob=applicant_dob
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Verification failure: {str(exc)}")

    visuals = raw_result.get("visuals", {})
    zoom_crops = visuals.get("zoom_crops", [])

    serialized_zooms = []
    for z in zoom_crops:
        serialized_zooms.append({
            "label": z.get("label", ""),
            "text": z.get("text", ""),
            "confidence": z.get("confidence", 0.0),
            "reasons": z.get("reasons", []),
            "patch_b64": bgr_to_base64(z.get("patch_bgr")),
            "ela_crop_b64": bgr_to_base64(z.get("ela_crop_bgr"))
        })

    serialized_visuals = {
        "aligned_image_b64": bgr_to_base64(visuals.get("aligned_bgr")),
        "alignment_overlay_b64": bgr_to_base64(visuals.get("alignment_overlay_bgr")),
        "segmentation_map_b64": bgr_to_base64(visuals.get("segmentation_map_bgr")),
        "ela_heatmap_b64": bgr_to_base64(visuals.get("ela_heatmap_bgr")),
        "verdict_overlay_b64": bgr_to_base64(visuals.get("verdict_overlay_bgr")),
        "zoom_crops": serialized_zooms
    }

    return {
        "verdict": raw_result["verdict"],
        "status": raw_result["status"],
        "is_authentic": raw_result["is_authentic"],
        "confidence_score": raw_result["confidence_score"],
        "processing_time_ms": raw_result["processing_time_ms"],
        "summary_message": raw_result["summary_message"],
        "five_check_stack": raw_result["five_check_stack"],
        "alignment_info": raw_result["alignment_info"],
        "counts": raw_result["counts"],
        "flagged_details": raw_result["flagged_details"],
        "visuals": serialized_visuals,
        "audit_report": raw_result["audit_report"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
