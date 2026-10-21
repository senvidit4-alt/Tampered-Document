# DocTamperGuard: Five-Check Verification & Anti-Tamper Engine

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=flat-square&logo=react)](https://vitejs.dev/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%20%2B%20ELA-5C3EE8?style=flat-square&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

**DocTamperGuard** is an industrial-grade, format-agnostic document authentication and digital forgery detection platform. It implements a multi-layered **Five-Check Verification Stack (Step 0 to Check 5)** capable of detecting micro-forgeries, pixel-level patch tampering, online PDF/web editor re-rasterizations (Sejda, Canva, Photopea, iLovePDF), Generative AI synthesis (Midjourney, DALL-E), unaccredited diploma mills, cryptographic QR inconsistencies, and cross-document applicant impersonation.

---

## 🏛️ Five-Check Verification Stack Architecture

```
                                [ Incoming Document (PDF / JPEG / PNG) ]
                                                   │
   ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
   │                                                                                               │
   ▼                                                                                               ▼
[ CHECK 1: File Forensics & Metadata ]                                          [ STEP 0: Intake & OCR Extraction ]
 • PDF Binary Stream Forensics (%%EOF > 1 revisions)                             • Multi-Line Header Parsing
 • Subsets & Font Injections (TrueType / Type0)                                  • Candidate Name, Roll No & DOB
 • Sejda, SAMBox, Cairo, Skia, Canva Signatures                                  • Course, Semester & Grade Mapping
 • Quantization & Stripped EXIF Cross-Signal                                                       │
   │                                                                                               ▼
   │                                                                            [ CHECK 2: Pixel Forensics ]
   │                                                                             • Error Level Analysis (ELA)
   │                                                                             • Sub-Pixel Baseline Drift
   │                                                                             • Digital Whiteout Box Detection
   │                                                                             • Font Ink & Sharpness Mismatch
   │                                                                                               │
   ├───────────────────────────────────────────────────────────────────────────────────────────────┤
   │                                                                                               │
   ▼                                                                                               ▼
[ CHECK 3: Cryptographic QR & Self-Consistency ]                                [ CHECK 4: Institutional Registry & Archive ]
 • Raw QR Parsing & Self-Consistency Verification                                • Global Accredited Registry (CBSE, UGC, RTU)
 • Digitally Signed Payload Integrity Check                                      • Central Archive Mock (DigiLocker Database)
 • Contradiction Detection vs Printed Text                                       • Diploma Mill & Blacklist Screening
   │                                                                                               │
   └───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                [ CHECK 5: Cross-Document Anti-Impersonation ]
                                 • Applicant Application vs Document Match
                                 • Name Distance & DOB Cross-Validation
                                                   │
                                                   ▼
                                [ Final Tri-State Verdict: PASS / REFER / FAIL ]
```

---

## 🎯 Key Detection Capabilities

### 1. Online PDF & Web Editors (Milestone 9)
- **Deep PDF Stream Forensics:** Inspects raw PDF dictionaries, `/Producer`, `/Creator`, `/Author`, and XMP metadata packets for signatures like `Sejda`, `SAMBox`, `Apache PDFBox`, `iText`, `Skia/PDF`, `Cairo`, `Canva`, and `Photopea`.
- **Incremental Save Detection:** Identifies multiple `%%EOF` markers and active `/Prev` cross-reference chains resulting from incremental online edits.
- **Injected Font Subsets:** Detects font subset injections (`[A-Z]{6}\+FontName`) and mixed `TrueType`/`Type0` fonts introduced when typing new text over original scans.
- **Digital Whiteout Patches (`DIGITAL_WHITEOUT_PATCH`):** Measures local micro-noise floors (`bg_noise_std`) and flat-gradient pixel ratios (`flat_bg_ratio`) to catch solid digital rectangles placed behind altered text.
- **Font Stroke Sharpness & Ink Mismatch (`FONT_INK_MISMATCH`):** Compares glyph 10th percentile ink luma (`p10_ink_luma`) and Sobel edge gradients against authentic scanned paper baseline.

### 2. Physical Pixel-Level Analysis (Check 2)
- **ELA Heatmaps:** Re-compresses documents to extract double-compression artifacts and compression energy discontinuities.
- **Sub-Pixel Spatial Drift:** Automatically segments table rows and computes per-cell baseline vertical drift (`delta-y` in pixels).

### 3. Institutional Legitimacy & Identity Verification (Checks 3, 4, 5)
- **Central Archive Lookup:** Verifies extracted student credentials against institutional registers.
- **Regulatory Screening:** Cross-references against comprehensive Indian School Boards, UGC-recognized Universities, and accredited International Institutions while catching blacklisted diploma mills.
- **Anti-Impersonation:** Ensures the verified document belongs to the actual applicant submitting the application.

---

## 📊 Benchmark Test Matrix

| Scenario | Target Document | Expected Verdict | Check Stack Status (C1 / C2 / C3 / C4 / C5) | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **0: Authentic Baseline** | Untouched Board Scan | **`PASS`** | `PASS / PASS / PASS / PASS / PASS` | **100%** |
| **1: Edited Genuine** | Modified Marks & Grades | **`FAIL`** | `FAIL / FAIL / FAIL / FAIL / PASS` | **100%** |
| **2: Fabricated Certificate** | Forged QR Signature | **`FAIL`** | `PASS / PASS / FAIL / FAIL / PASS` | **100%** |
| **3: Unrecognised Institution** | Blacklisted Diploma Mill | **`FAIL`** | `PASS / PASS / PASS / FAIL / FAIL` | **100%** |
| **4: Fully AI-Generated** | Midjourney / Stable Diffusion | **`FAIL`** | `FAIL / PASS / FAIL / REFER / FAIL` | **100%** |
| **5: Impersonation** | Genuine Scan, Wrong Person | **`FAIL`** | `PASS / PASS / PASS / PASS / FAIL` | **100%** |

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Clone & Set Up Backend

```bash
git clone https://github.com/senvidit4-alt/Tampered-Document.git
cd Tampered-Document

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI backend server
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be live at `http://127.0.0.1:8000` with Swagger docs at `http://127.0.0.1:8000/docs`.

### 2. Set Up Frontend

```bash
# In a new terminal window
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` to launch the interactive verification interface.

---

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, OpenCV, NumPy, EasyOCR, Pillow, PyPDF, PyPDFium2, qrcode.
- **Frontend:** React, Vite, Tailwind CSS, Lucide React icons.
- **Analysis:** Error Level Analysis (ELA), Sobel edge gradient variance, high-frequency background noise suppression, cryptographic signature validation.

---

## 📄 License
This project is licensed under the MIT License.
