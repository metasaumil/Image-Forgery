"""
classical_methods/metadata.py
------------------------------
EXIF Metadata Analysis for Forgery Detection.

THEORY:
Digital cameras embed metadata (EXIF data) in images: camera model, GPS,
datetime, software used. Forged images often show:
- Missing EXIF data (stripped by editing software)
- Inconsistent software tags (e.g., "Adobe Photoshop" vs camera EXIF)
- Mismatched timestamps
- GPS coordinates that don't match the claimed location
"""

import os
import json
from datetime import datetime
from PIL import Image
import piexif


# ─────────────────────────────────────────────
#  EXIF Tags of Interest
# ─────────────────────────────────────────────

SUSPICIOUS_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "paint", "affinity",
    "canva", "pixelmator", "corel", "snapseed", "facetune",
    "deepfake", "stable diffusion", "midjourney"
]

CAMERA_INDICATORS = [
    "Canon", "Nikon", "Sony", "Fujifilm", "Olympus",
    "Panasonic", "Samsung", "Apple", "Google"
]


# ─────────────────────────────────────────────
#  EXIF Extraction
# ─────────────────────────────────────────────

def extract_exif(image_path: str) -> dict:
    """
    Extract EXIF metadata from an image.

    Returns:
        Dictionary of all EXIF fields, or empty dict if none found.
    """
    result = {}

    try:
        img = Image.open(image_path)
        exif_data = img._getexif()  # Returns dict or None

        if exif_data is None:
            return {}

        # Map numeric tag IDs to human-readable names
        from PIL.ExifTags import TAGS
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            # Convert bytes to string for readability
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = str(value)
            result[tag_name] = value

    except Exception as e:
        result["_error"] = str(e)

    return result


# ─────────────────────────────────────────────
#  Forensic Analysis
# ─────────────────────────────────────────────

def analyze_metadata(image_path: str, verbose: bool = True) -> dict:
    """
    Forensic metadata analysis. Returns a suspicion report.

    Returns dict with:
        - exif_present: bool
        - suspicious_software: bool
        - editing_software: str or None
        - camera_make: str or None
        - camera_model: str or None
        - original_datetime: str or None
        - modification_datetime: str or None
        - datetime_inconsistency: bool
        - suspicion_score: float (0.0 to 1.0)
        - flags: list of suspicious findings
    """
    report = {
        "file": os.path.basename(image_path),
        "exif_present": False,
        "suspicious_software": False,
        "editing_software": None,
        "camera_make": None,
        "camera_model": None,
        "original_datetime": None,
        "modification_datetime": None,
        "datetime_inconsistency": False,
        "suspicion_score": 0.0,
        "flags": [],
        "raw_exif": {}
    }

    if not os.path.exists(image_path):
        report["flags"].append("File not found")
        return report

    exif = extract_exif(image_path)
    report["raw_exif"] = exif

    # ── Check if EXIF exists ──
    if not exif:
        report["flags"].append("No EXIF data found — may have been stripped by editing software")
        report["suspicion_score"] += 0.3
        if verbose:
            print("[Metadata] ⚠ No EXIF data. Score +0.3")
        return report  # Nothing more to analyze

    report["exif_present"] = True

    # ── Camera Info ──
    report["camera_make"]  = exif.get("Make")
    report["camera_model"] = exif.get("Model")

    # ── Software ──
    software = exif.get("Software", "")
    if software:
        software_lower = software.lower()
        report["editing_software"] = software
        for sus in SUSPICIOUS_SOFTWARE:
            if sus in software_lower:
                report["suspicious_software"] = True
                report["flags"].append(f"Editing software detected: {software}")
                report["suspicion_score"] += 0.4
                break

    # ── DateTime Analysis ──
    orig_dt = exif.get("DateTimeOriginal") or exif.get("DateTime")
    mod_dt  = exif.get("DateTimeDigitized") or exif.get("DateTime")

    report["original_datetime"]     = orig_dt
    report["modification_datetime"] = mod_dt

    if orig_dt and mod_dt and orig_dt != mod_dt:
        report["datetime_inconsistency"] = True
        report["flags"].append(f"Datetime inconsistency: original={orig_dt}, modified={mod_dt}")
        report["suspicion_score"] += 0.2

    # ── GPS in unexpected context ──
    if "GPSInfo" in exif and not report["camera_make"]:
        report["flags"].append("GPS present but no camera make — could be synthetic")
        report["suspicion_score"] += 0.1

    # ── No camera make despite claiming to be a photo ──
    if not report["camera_make"] and not report["editing_software"]:
        report["flags"].append("No camera make or software — unusual for real photos")
        report["suspicion_score"] += 0.15

    # Clip score to [0, 1]
    report["suspicion_score"] = min(1.0, report["suspicion_score"])

    if verbose:
        _print_report(report)

    return report


def _print_report(report: dict):
    print("\n" + "="*50)
    print(f"  METADATA ANALYSIS: {report['file']}")
    print("="*50)
    print(f"  EXIF Present:    {report['exif_present']}")
    print(f"  Camera:          {report['camera_make']} {report['camera_model']}")
    print(f"  Software:        {report['editing_software']}")
    print(f"  Orig DateTime:   {report['original_datetime']}")
    print(f"  Mod DateTime:    {report['modification_datetime']}")
    print(f"  Suspicion Score: {report['suspicion_score']:.2f}")
    if report["flags"]:
        print("\n  ⚠ FLAGS:")
        for f in report["flags"]:
            print(f"    → {f}")
    verdict = "⚠ SUSPICIOUS" if report["suspicion_score"] > 0.3 else "✓ LIKELY AUTHENTIC"
    print(f"\n  Verdict: {verdict}")
    print("="*50 + "\n")


def save_report(report: dict, save_path: str):
    """Save metadata report to a JSON file."""
    # Remove raw_exif to keep JSON clean (it can have non-serializable types)
    clean_report = {k: v for k, v in report.items() if k != "raw_exif"}
    with open(save_path, "w") as f:
        json.dump(clean_report, f, indent=2, default=str)
    print(f"[Metadata] Report saved to {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metadata.py <image_path>")
    else:
        report = analyze_metadata(sys.argv[1])
        save_report(report, "metadata_report.json")
