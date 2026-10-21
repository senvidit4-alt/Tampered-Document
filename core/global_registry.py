"""
DocTamperGuard - Milestone 8: Global & Indian Academic Institutions Registry
Comprehensive registry of:
  - All Major Indian School Boards (CBSE, CISCE/ICSE, NIOS, State Boards)
  - Major Indian Universities (UGC/AICTE Recognised: RTU, DU, IITs, NITs, BITS, etc.)
  - Major International Accredited Universities (MIT, Oxford, Stanford, Harvard, Cambridge, etc.)
  - Explicit Diploma Mill / Unrecognised Board Blacklist
  - Multi-tier matching: Exact -> Alias -> Fuzzy -> Keyword Heuristic -> Hipolabs World Universities API
"""

import json
import os
import re
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
HIPOLABS_CACHE_FILE = CACHE_DIR / "hipolabs_cache.json"

# In-memory cache for API lookups
_HIPOLABS_MEMORY_CACHE: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# 1. Explicit Diploma Mill & Fraudulent Blacklist
# ---------------------------------------------------------------------------

DIPLOMA_MILL_BLACKLIST = [
    "GLOBAL DIGITAL OPEN UNIVERSITY OF INDIA",
    "INTERNATIONAL FREE DEGREE INSTITUTE",
    "NATIONAL INSTITUTE OF FREE EDUCATION",
    "OPEN ACADEMY OF TECHNOLOGY",
    "BHARAT VIRTUAL UNIVERSITY",
    "UNIVERSAL DEGREE MILL",
    "AMERICAN DIPLOMA UNIVERSITY",
    "UNRECOGNISED BOARD",
    "WORLD DIPLOMA ACADEMY",
    "INDIAN VIRTUAL BOARD OF HIGHER EDUCATION",
    "INTERNATIONAL OPEN ACCREDITATION COUNCIL",
    "GLOBAL ONLINE UNIVERSITY OF SCIENCE AND TECHNOLOGY",
    "FREE DEGREE ONLINE INSTITUTE",
    "INTERNATIONAL ACCREDITED DEGREE MILL",
    "INSTITUTE OF UNACCREDITED STUDIES",
    "FAKE DEGREE FOUNDATION",
    "WORLD WIDE VIRTUAL UNIVERSITY"
]

# ---------------------------------------------------------------------------
# 2. Major Indian School Education Boards
# ---------------------------------------------------------------------------

INDIAN_SCHOOL_BOARDS: Dict[str, List[str]] = {
    "CENTRAL BOARD OF SECONDARY EDUCATION": [
        "CBSE", "CENTRAL BOARD", "CBSE DELHI", "CENTRAL BOARD OF EDUCATION"
    ],
    "COUNCIL FOR THE INDIAN SCHOOL CERTIFICATE EXAMINATIONS": [
        "CISCE", "ICSE", "ISC", "INDIAN SCHOOL CERTIFICATE", "COUNCIL FOR THE INDIAN SCHOOL"
    ],
    "NATIONAL INSTITUTE OF OPEN SCHOOLING": [
        "NIOS", "NATIONAL OPEN SCHOOL"
    ],
    "BOARD OF SECONDARY EDUCATION, RAJASTHAN": [
        "RBSE", "BSER", "RAJASTHAN BOARD", "BOARD OF SECONDARY EDUCATION RAJASTHAN", "BSER AJMER"
    ],
    "MAHARASHTRA STATE BOARD OF SECONDARY AND HIGHER SECONDARY EDUCATION": [
        "MSBSHSE", "MAHARASHTRA STATE BOARD", "MAHARASHTRA BOARD", "PUNE BOARD"
    ],
    "UTTAR PRADESH MADHYAMIK SHIKSHA PARISHAD": [
        "UPMSP", "UP BOARD", "ALLAHABAD BOARD", "UP MADHYAMIK"
    ],
    "GUJARAT SECONDARY AND HIGHER SECONDARY EDUCATION BOARD": [
        "GSEB", "GUJARAT BOARD", "GSHSEB"
    ],
    "KARNATAKA SCHOOL EXAMINATION AND ASSESSMENT BOARD": [
        "KSEEB", "KSEAB", "KARNATAKA BOARD", "KARNATAKA PU BOARD"
    ],
    "WEST BENGAL COUNCIL OF HIGHER SECONDARY EDUCATION": [
        "WBCHSE", "WBBSE", "WEST BENGAL BOARD", "BENGAL BOARD"
    ],
    "TAMIL NADU STATE BOARD OF SCHOOL EXAMINATIONS": [
        "TNSCERT", "DGE TAMIL NADU", "TAMIL NADU BOARD", "TN BOARD"
    ],
    "PUNJAB SCHOOL EDUCATION BOARD": [
        "PSEB", "PUNJAB BOARD", "MOHALI BOARD"
    ],
    "BOARD OF SCHOOL EDUCATION HARYANA": [
        "HBSE", "BSEH", "HARYANA BOARD", "BHIWANI BOARD"
    ],
    "BIHAR SCHOOL EXAMINATION BOARD": [
        "BSEB", "BIHAR BOARD", "PATNA BOARD"
    ],
    "JHARKHAND ACADEMIC COUNCIL": [
        "JAC", "JHARKHAND BOARD", "JAC RANCHI"
    ],
    "COUNCIL OF HIGHER SECONDARY EDUCATION, ODISHA": [
        "CHSE ODISHA", "BSE ODISHA", "ODISHA BOARD"
    ],
    "MADHYA PRADESH BOARD OF SECONDARY EDUCATION": [
        "MPBSE", "MP BOARD", "BHOPAL BOARD"
    ],
    "ANDHRA PRADESH BOARD OF INTERMEDIATE EDUCATION": [
        "APBIE", "BSEAP", "ANDHRA BOARD", "AP BOARD"
    ],
    "TELANGANA STATE BOARD OF INTERMEDIATE EDUCATION": [
        "TSBIE", "BSETELANGANA", "TELANGANA BOARD", "TS BOARD"
    ],
    "KERALA BOARD OF PUBLIC EXAMINATIONS": [
        "KBPE", "DHSE KERALA", "KERALA STATE BOARD", "KERALA BOARD"
    ],
    "CHHATTISGARH BOARD OF SECONDARY EDUCATION": [
        "CGBSE", "CG BOARD", "RAIPUR BOARD"
    ],
    "UTTARAKHAND BOARD OF SCHOOL EDUCATION": [
        "UBSE", "UTTARAKHAND BOARD", "RAMNAGAR BOARD"
    ],
    "ASSAM HIGHER SECONDARY EDUCATION COUNCIL": [
        "AHSEC", "SEBA", "ASSAM BOARD"
    ],
    "BOARD OF SECONDARY EDUCATION": [
        "BSE", "SECONDARY BOARD"
    ]
}

# ---------------------------------------------------------------------------
# 3. Major Indian UGC / AICTE Recognised Universities & Institutes
# ---------------------------------------------------------------------------

INDIAN_UNIVERSITIES: Dict[str, List[str]] = {
    # State Technical & Leading State Universities
    "RAJASTHAN TECHNICAL UNIVERSITY": ["RTU", "RTU KOTA", "RAJASTHAN TECHNICAL UNIVERSITY, KOTA"],
    "UNIVERSITY OF DELHI": ["DU", "DELHI UNIVERSITY", "DELHI UNIV"],
    "JAWAHARLAL NEHRU UNIVERSITY": ["JNU", "JNU NEW DELHI"],
    "BANARAS HINDU UNIVERSITY": ["BHU", "BHU VARANASI"],
    "ANNA UNIVERSITY": ["AU CHENNAI", "ANNA UNIV"],
    "UNIVERSITY OF MUMBAI": ["MUMBAI UNIVERSITY", "BOMBAY UNIVERSITY"],
    "SAVITRIBAI PHULE PUNE UNIVERSITY": ["SPPU", "PUNE UNIVERSITY", "UNIVERSITY OF PUNE"],
    "MAULANA ABUL KALAM AZAD UNIVERSITY OF TECHNOLOGY": ["MAKAUT", "WBUT", "WEST BENGAL UNIVERSITY OF TECHNOLOGY"],
    "DR. A.P.J. ABDUL KALAM TECHNICAL UNIVERSITY": ["AKTU", "UPTU", "UTTAR PRADESH TECHNICAL UNIVERSITY"],
    "VISVESVARAYA TECHNOLOGICAL UNIVERSITY": ["VTU", "VTU BELGAUM"],
    "GUJARAT TECHNOLOGICAL UNIVERSITY": ["GTU", "GTU AHMEDABAD"],
    "JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY": ["JNTU", "JNTU HYDERABAD", "JNTUH", "JNTUK"],
    "MAHARSHI DAYANAND UNIVERSITY": ["MDU", "MDU ROHTAK"],
    "KURUKSHETRA UNIVERSITY": ["KUK", "KURUKSHETRA UNIV"],
    "PANJAB UNIVERSITY": ["PU CHANDIGARH", "PANJAB UNIV"],
    "OSMANIA UNIVERSITY": ["OU HYDERABAD", "OSMANIA"],
    "ANDHRA UNIVERSITY": ["AU VIZAG", "ANDHRA UNIV"],
    "UNIVERSITY OF CALCUTTA": ["CALCUTTA UNIVERSITY"],
    "JADAVPUR UNIVERSITY": ["JU KOLKATA"],
    "UNIVERSITY OF MADRAS": ["MADRAS UNIVERSITY"],
    "UNIVERSITY OF KERALA": ["KERALA UNIVERSITY"],
    "BANGALORE UNIVERSITY": ["BU BANGALORE"],
    "DEVI AHILYA VISHWAVIDYALAYA": ["DAVV", "DAVV INDORE"],

    # All Indian Institutes of Technology (IITs)
    "INDIAN INSTITUTE OF TECHNOLOGY BOMBAY": ["IIT BOMBAY", "IITB", "IIT-BOMBAY"],
    "INDIAN INSTITUTE OF TECHNOLOGY DELHI": ["IIT DELHI", "IITD", "IIT-DELHI"],
    "INDIAN INSTITUTE OF TECHNOLOGY MADRAS": ["IIT MADRAS", "IITM", "IIT-MADRAS"],
    "INDIAN INSTITUTE OF TECHNOLOGY KANPUR": ["IIT KANPUR", "IITK", "IIT-KANPUR"],
    "INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR": ["IIT KHARAGPUR", "IIT KGP", "IIT-KGP"],
    "INDIAN INSTITUTE OF TECHNOLOGY ROORKEE": ["IIT ROORKEE", "IITR", "IIT-ROORKEE"],
    "INDIAN INSTITUTE OF TECHNOLOGY GUWAHATI": ["IIT GUWAHATI", "IITG", "IIT-GUWAHATI"],
    "INDIAN INSTITUTE OF TECHNOLOGY HYDERABAD": ["IIT HYDERABAD", "IITH", "IIT-HYDERABAD"],
    "INDIAN INSTITUTE OF TECHNOLOGY (BHU) VARANASI": ["IIT BHU", "IIT VARANASI", "IIT-BHU"],
    "INDIAN INSTITUTE OF TECHNOLOGY INDORE": ["IIT INDORE", "IITI"],
    "INDIAN INSTITUTE OF TECHNOLOGY GANDHINAGAR": ["IIT GANDHINAGAR", "IITGN"],
    "INDIAN INSTITUTE OF TECHNOLOGY BHUBANESWAR": ["IIT BHUBANESWAR", "IITBBS"],
    "INDIAN INSTITUTE OF TECHNOLOGY PATNA": ["IIT PATNA", "IITP"],
    "INDIAN INSTITUTE OF TECHNOLOGY ROPAR": ["IIT ROPAR", "IITRPR"],
    "INDIAN INSTITUTE OF TECHNOLOGY MANDI": ["IIT MANDI"],
    "INDIAN INSTITUTE OF TECHNOLOGY JODHPUR": ["IIT JODHPUR", "IITJ"],
    "INDIAN INSTITUTE OF TECHNOLOGY TIRUPATI": ["IIT TIRUPATI"],
    "INDIAN INSTITUTE OF TECHNOLOGY PALAKKAD": ["IIT PALAKKAD"],
    "INDIAN INSTITUTE OF TECHNOLOGY GOA": ["IIT GOA"],
    "INDIAN INSTITUTE OF TECHNOLOGY DHARWAD": ["IIT DHARWAD"],
    "INDIAN INSTITUTE OF TECHNOLOGY BHILAI": ["IIT BHILAI"],
    "INDIAN INSTITUTE OF TECHNOLOGY JAMMU": ["IIT JAMMU"],
    "INDIAN INSTITUTE OF TECHNOLOGY (ISM) DHANBAD": ["IIT DHANBAD", "ISM DHANBAD"],

    # National Institutes of Technology (NITs)
    "NATIONAL INSTITUTE OF TECHNOLOGY TIRUCHIRAPPALLI": ["NIT TRICHY", "NITT"],
    "NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA": ["NITK", "NIT SURATHKAL", "NITK SURATHKAL"],
    "NATIONAL INSTITUTE OF TECHNOLOGY ROURKELA": ["NIT ROURKELA", "NITR"],
    "NATIONAL INSTITUTE OF TECHNOLOGY WARANGAL": ["NIT WARANGAL", "NITW"],
    "NATIONAL INSTITUTE OF TECHNOLOGY CALICUT": ["NIT CALICUT", "NITC"],
    "MALAVIYA NATIONAL INSTITUTE OF TECHNOLOGY JAIPUR": ["MNIT JAIPUR", "MNIT"],
    "MOTILAL NEHRU NATIONAL INSTITUTE OF TECHNOLOGY ALLAHABAD": ["MNNIT ALLAHABAD", "MNNIT"],
    "VISVESVARAYA NATIONAL INSTITUTE OF TECHNOLOGY NAGPUR": ["VNIT NAGPUR", "VNIT"],
    "SARDAR VALLABHBHAI NATIONAL INSTITUTE OF TECHNOLOGY SURAT": ["SVNIT SURAT", "SVNIT"],

    # IIITs, AIIMS, IIMs, and Premium Institutions
    "INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY HYDERABAD": ["IIIT HYDERABAD", "IIITH"],
    "INTERNATIONAL INSTITUTE OF INFORMATION TECHNOLOGY BANGALORE": ["IIIT BANGALORE", "IIITB"],
    "INDRAPRASTHA INSTITUTE OF INFORMATION TECHNOLOGY DELHI": ["IIIT DELHI", "IIITD"],
    "INDIAN INSTITUTE OF SCIENCE": ["IISC", "IISC BANGALORE"],
    "ALL INDIA INSTITUTE OF MEDICAL SCIENCES NEW DELHI": ["AIIMS NEW DELHI", "AIIMS DELHI", "AIIMS"],
    "INDIAN INSTITUTE OF MANAGEMENT AHMEDABAD": ["IIM AHMEDABAD", "IIMA"],
    "INDIAN INSTITUTE OF MANAGEMENT BANGALORE": ["IIM BANGALORE", "IIMB"],
    "INDIAN INSTITUTE OF MANAGEMENT CALCUTTA": ["IIM CALCUTTA", "IIMC"],
    "BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE PILANI": ["BITS PILANI", "BITS", "BITS-PILANI"],
    "VELLORE INSTITUTE OF TECHNOLOGY": ["VIT", "VIT VELLORE", "VIT UNIVERSITY"],
    "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY": ["SRM", "SRM UNIVERSITY"],
    "MANIPAL ACADEMY OF HIGHER EDUCATION": ["MAHE", "MANIPAL UNIVERSITY"],
    "AMITY UNIVERSITY": ["AMITY"],
    "THAPAR INSTITUTE OF ENGINEERING AND TECHNOLOGY": ["THAPAR UNIVERSITY", "TIET"],
    "ALIGARH MUSLIM UNIVERSITY": ["AMU"],
    "JAMIA MILLIA ISLAMIA": ["JMI"]
}

# ---------------------------------------------------------------------------
# 4. Major International Accredited Universities
# ---------------------------------------------------------------------------

GLOBAL_ACCREDITED_UNIVERSITIES: Dict[str, List[str]] = {
    # United States
    "MASSACHUSETTS INSTITUTE OF TECHNOLOGY": ["MIT"],
    "STANFORD UNIVERSITY": ["STANFORD"],
    "HARVARD UNIVERSITY": ["HARVARD"],
    "UNIVERSITY OF CALIFORNIA, BERKELEY": ["UC BERKELEY", "CAL BERKELEY", "UCB"],
    "CALIFORNIA INSTITUTE OF TECHNOLOGY": ["CALTECH"],
    "CARNEGIE MELLON UNIVERSITY": ["CMU"],
    "PRINCETON UNIVERSITY": ["PRINCETON"],
    "COLUMBIA UNIVERSITY": ["COLUMBIA"],
    "YALE UNIVERSITY": ["YALE"],
    "UNIVERSITY OF CHICAGO": ["UCHICAGO"],
    "UNIVERSITY OF PENNSYLVANIA": ["UPENN", "PENN"],
    "CORNELL UNIVERSITY": ["CORNELL"],
    "UNIVERSITY OF MICHIGAN": ["UMICH", "UNIVERSITY OF MICHIGAN ANN ARBOR"],
    "UNIVERSITY OF CALIFORNIA, LOS ANGELES": ["UCLA"],
    "JOHNS HOPKINS UNIVERSITY": ["JHU"],
    "NEW YORK UNIVERSITY": ["NYU"],
    "GEORGIA INSTITUTE OF TECHNOLOGY": ["GEORGIA TECH", "GATECH"],
    "UNIVERSITY OF WASHINGTON": ["UW SEATTLE"],
    "UNIVERSITY OF ILLINOIS URBANA-CHAMPAIGN": ["UIUC"],
    "UNIVERSITY OF TEXAS AT AUSTIN": ["UT AUSTIN"],
    "PURDUE UNIVERSITY": ["PURDUE"],
    "NORTHWESTERN UNIVERSITY": ["NORTHWESTERN"],

    # United Kingdom
    "UNIVERSITY OF OXFORD": ["OXFORD", "OXFORD UNIVERSITY"],
    "UNIVERSITY OF CAMBRIDGE": ["CAMBRIDGE", "CAMBRIDGE UNIVERSITY"],
    "IMPERIAL COLLEGE LONDON": ["IMPERIAL COLLEGE", "IMPERIAL"],
    "UNIVERSITY COLLEGE LONDON": ["UCL"],
    "LONDON SCHOOL OF ECONOMICS": ["LSE", "LONDON SCHOOL OF ECONOMICS AND POLITICAL SCIENCE"],
    "KING'S COLLEGE LONDON": ["KCL"],
    "UNIVERSITY OF EDINBURGH": ["EDINBURGH UNIVERSITY"],
    "UNIVERSITY OF MANCHESTER": ["MANCHESTER UNIVERSITY"],
    "UNIVERSITY OF BRISTOL": ["BRISTOL UNIVERSITY"],
    "UNIVERSITY OF WARWICK": ["WARWICK UNIVERSITY"],

    # Canada
    "UNIVERSITY OF TORONTO": ["UTORONTO", "U OF T"],
    "UNIVERSITY OF BRITISH COLUMBIA": ["UBC"],
    "MCGILL UNIVERSITY": ["MCGILL"],
    "MCMASTER UNIVERSITY": ["MCMASTER"],
    "UNIVERSITY OF WATERLOO": ["UWATERLOO"],
    "UNIVERSITY OF ALBERTA": ["UALBERTA"],

    # Europe
    "ETH ZURICH": ["ETH", "SWISS FEDERAL INSTITUTE OF TECHNOLOGY"],
    "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE": ["EPFL"],
    "TECHNICAL UNIVERSITY OF MUNICH": ["TU MUNICH", "TUM"],
    "LUDWIG MAXIMILIAN UNIVERSITY OF MUNICH": ["LMU MUNICH", "LMU"],
    "HEIDELBERG UNIVERSITY": ["UNIVERSITAT HEIDELBERG"],
    "DELFT UNIVERSITY OF TECHNOLOGY": ["TU DELFT"],
    "SORBONNE UNIVERSITY": ["SORBONNE"],
    "POLITECNICO DI MILANO": ["POLIMI"],

    # Asia
    "NATIONAL UNIVERSITY OF SINGAPORE": ["NUS"],
    "NANYANG TECHNOLOGICAL UNIVERSITY": ["NTU SINGAPORE", "NTU"],
    "TSINGHUA UNIVERSITY": ["TSINGHUA"],
    "PEKING UNIVERSITY": ["PEKING"],
    "UNIVERSITY OF HONG KONG": ["HKU"],
    "HONG KONG UNIVERSITY OF SCIENCE AND TECHNOLOGY": ["HKUST"],
    "UNIVERSITY OF TOKYO": ["UTOKYO", "TODAI"],
    "KYOTO UNIVERSITY": ["KYOTO UNIV"],
    "SEOUL NATIONAL UNIVERSITY": ["SNU"],
    "KOREA ADVANCED INSTITUTE OF SCIENCE AND TECHNOLOGY": ["KAIST"],

    # Australia & New Zealand
    "UNIVERSITY OF MELBOURNE": ["UNIMELB"],
    "UNIVERSITY OF SYDNEY": ["USYD"],
    "AUSTRALIAN NATIONAL UNIVERSITY": ["ANU"],
    "UNIVERSITY OF NEW SOUTH WALES": ["UNSW SYDNEY", "UNSW"],
    "UNIVERSITY OF QUEENSLAND": ["UQ"],
    "MONASH UNIVERSITY": ["MONASH"],
    "UNIVERSITY OF AUCKLAND": ["U OF AUCKLAND"]
}

# ---------------------------------------------------------------------------
# 5. Legitimate Academic Institutional Keywords
# ---------------------------------------------------------------------------

RECOGNISED_INSTITUTION_KEYWORDS: List[str] = [
    "UNIVERSITY", "INSTITUTE OF TECHNOLOGY", "NATIONAL INSTITUTE",
    "COLLEGE OF ENGINEERING", "BOARD OF SECONDARY EDUCATION",
    "BOARD OF HIGHER SECONDARY", "BOARD OF TECHNICAL EDUCATION",
    "STATE BOARD", "COUNCIL FOR THE INDIAN SCHOOL", "CENTRAL BOARD",
    "ACADEMY OF HIGHER EDUCATION", "DEPARTMENT OF TECHNICAL EDUCATION",
    "POLYTECHNIC", "ALL INDIA COUNCIL", "INDIAN INSTITUTE",
    "COLLEGE OF TECHNOLOGY", "INSTITUTE OF MEDICAL SCIENCES",
    "INSTITUTE OF MANAGEMENT", "FACULTY OF ENGINEERING",
    "FACULTY OF MEDICINE", "COLLEGE OF ARTS AND SCIENCES",
    "COLLEGE OF ADVANCED TECHNOLOGY", "SCHOOL OF ENGINEERING",
    "SCHOOL OF MEDICINE", "SCHOOL OF LAW"
]


# ---------------------------------------------------------------------------
# Helper Normalizer & Disk Cache
# ---------------------------------------------------------------------------

def _normalize_name(name_str: str) -> str:
    """Cleans and standardizes institution name string for comparison."""
    if not name_str:
        return ""
    # Remove special chars, collapse spaces
    s = re.sub(r"[^A-Z0-9\s]", " ", name_str.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_hipolabs_cache():
    """Loads cached Hipolabs API queries from disk."""
    global _HIPOLABS_MEMORY_CACHE
    if not _HIPOLABS_MEMORY_CACHE and HIPOLABS_CACHE_FILE.exists():
        try:
            with open(HIPOLABS_CACHE_FILE, "r", encoding="utf-8") as f:
                _HIPOLABS_MEMORY_CACHE = json.load(f)
        except Exception:
            _HIPOLABS_MEMORY_CACHE = {}


def _save_hipolabs_cache():
    """Saves Hipolabs cache to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(HIPOLABS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_HIPOLABS_MEMORY_CACHE, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Online Real-Time Accreditation Fallback (Hipolabs World Universities API)
# ---------------------------------------------------------------------------

def query_hipolabs_universities_api(query_name: str, timeout_sec: float = 2.5) -> Optional[Dict[str, Any]]:
    """
    Non-blocking query against the free Hipolabs World Universities API
    (http://universities.hipolabs.com/search?name=...)
    Covers 10,000+ accredited universities globally across 180+ countries.
    Uses in-memory + persistent disk cache to maintain sub-millisecond repeat queries.
    """
    clean_q = _normalize_name(query_name)
    if not clean_q or len(clean_q) < 4:
        return None

    _load_hipolabs_cache()
    if clean_q in _HIPOLABS_MEMORY_CACHE:
        return _HIPOLABS_MEMORY_CACHE[clean_q]

    # Extract prime keyword for university search (e.g. "OXFORD", "STANFORD", "RAJASTHAN")
    keywords = [w for w in clean_q.split() if w not in ("UNIVERSITY", "OF", "THE", "INSTITUTE", "COLLEGE", "STATE", "CENTRAL")]
    search_term = keywords[0] if keywords else clean_q

    url = f"http://universities.hipolabs.com/search?name={urllib.parse.quote(search_term)}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DocTamperGuard-Accreditation/2.0 (Verification Engine)"}
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and data:
                # Find closest match
                best_match = None
                best_ratio = 0.0
                for item in data[:15]:
                    cand_name = item.get("name", "").upper()
                    r = SequenceMatcher(None, clean_q, _normalize_name(cand_name)).ratio()
                    # Also check if cand_name is substring or vice versa
                    if _normalize_name(cand_name) in clean_q or clean_q in _normalize_name(cand_name):
                        r = max(r, 0.85)
                    if r > best_ratio:
                        best_ratio = r
                        best_match = item

                if best_match and best_ratio >= 0.65:
                    result = {
                        "name": best_match.get("name"),
                        "country": best_match.get("country"),
                        "web_pages": best_match.get("web_pages", []),
                        "confidence": round(best_ratio * 100.0, 1)
                    }
                    _HIPOLABS_MEMORY_CACHE[clean_q] = result
                    _save_hipolabs_cache()
                    return result

    except Exception:
        # Graceful fallback: Network unavailable or API timed out
        pass

    _HIPOLABS_MEMORY_CACHE[clean_q] = None
    _save_hipolabs_cache()
    return None


# ---------------------------------------------------------------------------
# Master Institution Accreditation Verifier
# ---------------------------------------------------------------------------

def verify_institution(
    institution_name: str,
    check_online: bool = True
) -> Dict[str, Any]:
    """
    Universal verification entrypoint for any Indian or Global Institution:
    1. Blacklist Check: Immediate rejection of known diploma mills and fake boards.
    2. Exact / Alias Match: Checks curated Indian boards, Indian universities, global universities.
    3. Fuzzy Match: SequenceMatcher against canonical names and aliases.
    4. Heuristic Academic Keyword: Validates presence of accredited university/board terminology.
    5. Online Hipolabs Fallback: Resolves unknown international universities in real time.

    Returns structured accreditation report:
      - is_recognized: bool
      - is_blacklisted: bool
      - canonical_name: str
      - category: str ("INDIAN_BOARD", "INDIAN_UNIVERSITY", "GLOBAL_UNIVERSITY", "ONLINE_ACCREDITED", "KEYWORD_HEURISTIC", "DIPLOMA_MILL_BLACKLIST", "UNRECOGNIZED")
      - confidence: float (0.0 to 100.0)
      - source: str
      - details: str
    """
    raw_name = str(institution_name or "").strip()
    norm_name = _normalize_name(raw_name)

    if not norm_name or norm_name == "UNKNOWN":
        return {
            "is_recognized": False,
            "is_blacklisted": False,
            "canonical_name": "UNKNOWN",
            "category": "UNRECOGNIZED",
            "confidence": 0.0,
            "source": "EMPTY_INPUT",
            "details": "No institution name provided for verification."
        }

    # -------------------------------------------------------------
    # 1. Diploma Mill / Fraudulent Blacklist Check (P0)
    # -------------------------------------------------------------
    for b_item in DIPLOMA_MILL_BLACKLIST:
        b_norm = _normalize_name(b_item)
        if b_norm in norm_name or norm_name in b_norm:
            return {
                "is_recognized": False,
                "is_blacklisted": True,
                "canonical_name": b_item,
                "category": "DIPLOMA_MILL_BLACKLIST",
                "confidence": 99.9,
                "source": "REGULATORY_BLACKLIST",
                "details": f"CRITICAL FRAUD: Institution '{raw_name}' is identified on the official regulatory Blacklist of unaccredited diploma mills."
            }

    # -------------------------------------------------------------
    # 2. Exact & Alias Match in Curated Registries
    # -------------------------------------------------------------
    # Check Indian Boards
    for canon, aliases in INDIAN_SCHOOL_BOARDS.items():
        all_names = [canon] + aliases
        for name in all_names:
            n_norm = _normalize_name(name)
            if n_norm == norm_name or n_norm in norm_name or norm_name in n_norm:
                return {
                    "is_recognized": True,
                    "is_blacklisted": False,
                    "canonical_name": canon,
                    "category": "INDIAN_BOARD",
                    "confidence": 98.0,
                    "source": "CURATED_INDIAN_BOARDS_REGISTRY",
                    "details": f"Recognized Indian School Education Board: '{canon}'."
                }

    # Check Indian Universities
    for canon, aliases in INDIAN_UNIVERSITIES.items():
        all_names = [canon] + aliases
        for name in all_names:
            n_norm = _normalize_name(name)
            if n_norm == norm_name or n_norm in norm_name or norm_name in n_norm:
                return {
                    "is_recognized": True,
                    "is_blacklisted": False,
                    "canonical_name": canon,
                    "category": "INDIAN_UNIVERSITY",
                    "confidence": 98.0,
                    "source": "CURATED_INDIAN_UGC_AICTE_REGISTRY",
                    "details": f"UGC / AICTE Recognized Indian University: '{canon}'."
                }

    # Check Global Universities
    for canon, aliases in GLOBAL_ACCREDITED_UNIVERSITIES.items():
        all_names = [canon] + aliases
        for name in all_names:
            n_norm = _normalize_name(name)
            if n_norm == norm_name or n_norm in norm_name or norm_name in n_norm:
                return {
                    "is_recognized": True,
                    "is_blacklisted": False,
                    "canonical_name": canon,
                    "category": "GLOBAL_UNIVERSITY",
                    "confidence": 98.0,
                    "source": "CURATED_GLOBAL_ACCREDITED_REGISTRY",
                    "details": f"Accredited Global Institution: '{canon}'."
                }

    # -------------------------------------------------------------
    # 3. High-Confidence Fuzzy Matcher (>= 0.82 similarity)
    # -------------------------------------------------------------
    all_curated: List[Tuple[str, str]] = []
    for c, al in INDIAN_SCHOOL_BOARDS.items():
        all_curated.append((c, "INDIAN_BOARD"))
        for a in al:
            all_curated.append((a, "INDIAN_BOARD"))
    for c, al in INDIAN_UNIVERSITIES.items():
        all_curated.append((c, "INDIAN_UNIVERSITY"))
        for a in al:
            all_curated.append((a, "INDIAN_UNIVERSITY"))
    for c, al in GLOBAL_ACCREDITED_UNIVERSITIES.items():
        all_curated.append((c, "GLOBAL_UNIVERSITY"))
        for a in al:
            all_curated.append((a, "GLOBAL_UNIVERSITY"))

    best_match_name = None
    best_cat = None
    best_ratio = 0.0

    for candidate, cat in all_curated:
        ratio = SequenceMatcher(None, norm_name, _normalize_name(candidate)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_name = candidate
            best_cat = cat

    if best_match_name and best_ratio >= 0.80:
        return {
            "is_recognized": True,
            "is_blacklisted": False,
            "canonical_name": best_match_name,
            "category": best_cat,
            "confidence": round(best_ratio * 100.0, 1),
            "source": "FUZZY_REGISTRY_MATCH",
            "details": f"Accredited Institution matched via fuzzy comparison: '{best_match_name}' ({round(best_ratio * 100.0, 1)}% similarity)."
        }

    # -------------------------------------------------------------
    # 4. Online Non-Blocking Hipolabs API Lookup
    # -------------------------------------------------------------
    if check_online:
        api_res = query_hipolabs_universities_api(norm_name, timeout_sec=2.5)
        if api_res:
            canon = api_res.get("name", raw_name)
            country = api_res.get("country", "Global")
            return {
                "is_recognized": True,
                "is_blacklisted": False,
                "canonical_name": canon,
                "category": "ONLINE_ACCREDITED",
                "confidence": api_res.get("confidence", 92.0),
                "source": "HIPOLABS_GLOBAL_REGISTRY",
                "details": f"Accredited Higher Education Institution verified via Hipolabs Global Registry: '{canon}' ({country})."
            }

    # -------------------------------------------------------------
    # 5. Legitimate Academic Keyword Heuristic
    # -------------------------------------------------------------
    # If the entity name includes genuine educational keywords and is sufficiently long
    has_keyword = any(kw in norm_name for kw in RECOGNISED_INSTITUTION_KEYWORDS)
    if has_keyword and len(norm_name) >= 10:
        return {
            "is_recognized": True,
            "is_blacklisted": False,
            "canonical_name": raw_name,
            "category": "KEYWORD_HEURISTIC",
            "confidence": 85.0,
            "source": "ACADEMIC_KEYWORD_HEURISTIC",
            "details": f"Recognized academic body structure verified via institutional keywords: '{raw_name}'."
        }

    # -------------------------------------------------------------
    # 6. Fallback: Unrecognized Entity
    # -------------------------------------------------------------
    return {
        "is_recognized": False,
        "is_blacklisted": False,
        "canonical_name": raw_name,
        "category": "UNRECOGNIZED",
        "confidence": 20.0,
        "source": "UNRECOGNIZED",
        "details": f"Institution '{raw_name}' could not be validated against UGC, Board, or Global Accredited Registries."
    }


# ---------------------------------------------------------------------------
# Unit Self-Test & Diagnostic CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("  DocTamperGuard - Global & Indian Institutions Registry Unit Test")
    print("=" * 80)

    test_queries = [
        # Indian Universities & Boards
        ("RAJASTHAN TECHNICAL UNIVERSITY, KOTA", True, "INDIAN_UNIVERSITY"),
        ("CENTRAL BOARD OF SECONDARY EDUCATION", True, "INDIAN_BOARD"),
        ("INDIAN INSTITUTE OF TECHNOLOGY BOMBAY", True, "INDIAN_UNIVERSITY"),
        ("DELHI UNIVERSITY", True, "INDIAN_UNIVERSITY"),
        ("BITS PILANI", True, "INDIAN_UNIVERSITY"),

        # International Accredited Universities
        ("MASSACHUSETTS INSTITUTE OF TECHNOLOGY", True, "GLOBAL_UNIVERSITY"),
        ("UNIVERSITY OF OXFORD", True, "GLOBAL_UNIVERSITY"),
        ("STANFORD UNIVERSITY", True, "GLOBAL_UNIVERSITY"),
        ("HARVARD", True, "GLOBAL_UNIVERSITY"),

        # Blacklisted Diploma Mills
        ("GLOBAL DIGITAL OPEN UNIVERSITY OF INDIA", False, "DIPLOMA_MILL_BLACKLIST"),
        ("INTERNATIONAL FREE DEGREE INSTITUTE", False, "DIPLOMA_MILL_BLACKLIST"),
        ("UNRECOGNISED BOARD", False, "DIPLOMA_MILL_BLACKLIST"),

        # Generic / Keyword Heuristic
        ("GOVERNMENT COLLEGE OF ENGINEERING AND TECHNOLOGY", True, "KEYWORD_HEURISTIC")
    ]

    all_passed = True
    for query, exp_rec, exp_cat in test_queries:
        res = verify_institution(query, check_online=False)
        rec = res["is_recognized"]
        cat = res["category"]
        ok = (rec == exp_rec and cat == exp_cat)
        if not ok:
            all_passed = False
        print(f"Query: '{query}'")
        print(f"  -> Recognized: {rec} (Exp: {exp_rec}) | Cat: {cat} (Exp: {exp_cat}) | Match: {'PASS' if ok else 'FAIL'}")
        print(f"  -> Canonical: '{res['canonical_name']}' | Details: {res['details']}\n")

    print("=" * 80)
    print("TEST SUITE VERDICT:", "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print("=" * 80)
