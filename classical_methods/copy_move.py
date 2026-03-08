"""
classical_methods/copy_move.py
--------------------------------
Copy-Move Forgery Detection using ORB/SIFT feature matching.

THEORY:
Copy-move forgery: a region of an image is copied and pasted elsewhere
within the same image (to hide or duplicate something).

HOW IT WORKS:
1. Extract keypoints from the image using ORB (or SIFT)
2. Find similar feature pairs within the same image
3. If many matched keypoint pairs are far apart spatially but very similar,
   it suggests a region was copied from one location to another.

ORB (Oriented FAST and Rotated BRIEF):
- Fast and free (unlike SIFT which was patented)
- Rotation invariant
- Good enough for most copy-move cases

SIFT:
- More accurate but slower
- Requires opencv-contrib-python
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os


# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────

def extract_keypoints(image: np.ndarray, method: str = "ORB", max_features: int = 5000):
    """
    Extract keypoints and descriptors from an image.

    Args:
        image:        BGR or grayscale numpy array
        method:       'ORB' (fast, free) or 'SIFT' (accurate, needs contrib)
        max_features: Maximum number of keypoints to detect

    Returns:
        (keypoints, descriptors)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features)
        kp, desc = detector.detectAndCompute(gray, None)
    elif method == "SIFT":
        try:
            detector = cv2.SIFT_create(nfeatures=max_features)
            kp, desc = detector.detectAndCompute(gray, None)
        except AttributeError:
            print("[WARNING] SIFT not available. Using ORB instead.")
            print("         Install: pip install opencv-contrib-python")
            detector = cv2.ORB_create(nfeatures=max_features)
            kp, desc = detector.detectAndCompute(gray, None)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ORB' or 'SIFT'.")

    return kp, desc


# ─────────────────────────────────────────────
#  Matching
# ─────────────────────────────────────────────

def match_features(desc1: np.ndarray, method: str = "ORB",
                   ratio_threshold: float = 0.75) -> List:
    """
    Match descriptors within the SAME image (self-matching).
    Filters matches using Lowe's ratio test to remove bad matches.

    Args:
        desc1:           Descriptors array
        method:          'ORB' uses Hamming distance, 'SIFT' uses L2
        ratio_threshold: Lowe's ratio test threshold (lower = stricter)

    Returns:
        List of good DMatch objects
    """
    if desc1 is None or len(desc1) < 2:
        return []

    norm_type = cv2.NORM_HAMMING if method == "ORB" else cv2.NORM_L2
    matcher   = cv2.BFMatcher(norm_type, crossCheck=False)

    # knnMatch returns 2 nearest neighbors for ratio test
    matches = matcher.knnMatch(desc1, desc1, k=3)

    good_matches = []
    for match_group in matches:
        if len(match_group) < 3:
            continue
        # match_group[0] is always self (distance=0), skip it
        # Compare 1st vs 2nd non-self match
        m, n = match_group[1], match_group[2]
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches


# ─────────────────────────────────────────────
#  Spatial Filtering
# ─────────────────────────────────────────────

def filter_by_distance(matches: List, keypoints: tuple,
                        min_dist_px: float = 30.0) -> List:
    """
    Remove matches where the two keypoints are too close spatially.
    Genuine copy-move regions are separated in space.

    Args:
        matches:       List of DMatch objects
        keypoints:     Tuple of keypoints
        min_dist_px:   Minimum pixel distance between matched points

    Returns:
        Filtered list of matches
    """
    filtered = []
    for m in matches:
        pt1 = np.array(keypoints[m.queryIdx].pt)
        pt2 = np.array(keypoints[m.trainIdx].pt)
        dist = np.linalg.norm(pt1 - pt2)
        if dist > min_dist_px:
            filtered.append(m)
    return filtered


# ─────────────────────────────────────────────
#  Main Detection Pipeline
# ─────────────────────────────────────────────

def detect_copy_move(image_path: str, method: str = "ORB",
                     min_matches: int = 10,
                     save_path: str = None,
                     show: bool = True) -> Tuple[bool, int, np.ndarray]:
    """
    Full copy-move detection pipeline.

    Args:
        image_path:  Path to image
        method:      'ORB' or 'SIFT'
        min_matches: Minimum suspicious match pairs to flag as forged
        save_path:   Save visualization to this path
        show:        Display visualization

    Returns:
        (is_forged: bool, num_matches: int, visualization: np.ndarray)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Step 1: Extract features
    keypoints, descriptors = extract_keypoints(image, method)
    print(f"[Copy-Move] Detected {len(keypoints)} keypoints")

    if descriptors is None or len(keypoints) < 10:
        print("[Copy-Move] Too few keypoints. Image may be too simple.")
        return False, 0, image

    # Step 2: Self-match
    matches = match_features(descriptors, method)
    print(f"[Copy-Move] Raw matches after ratio test: {len(matches)}")

    # Step 3: Filter by spatial distance
    matches = filter_by_distance(matches, keypoints, min_dist_px=30)
    print(f"[Copy-Move] Matches after spatial filter: {len(matches)}")

    # Step 4: Decision
    is_forged = len(matches) >= min_matches

    # Step 5: Visualization
    vis = _visualize(image, keypoints, matches, is_forged)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[Copy-Move] Saved to {save_path}")

    if show:
        plt.figure(figsize=(12, 6))
        plt.imshow(vis)
        verdict = "⚠ COPY-MOVE DETECTED" if is_forged else "✓ NO COPY-MOVE FOUND"
        plt.title(f"{verdict} | Suspicious matches: {len(matches)}", fontsize=13)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return is_forged, len(matches), vis


def _visualize(image: np.ndarray, keypoints, matches: List, is_forged: bool) -> np.ndarray:
    """
    Draw lines between matched keypoint pairs on the image.
    Green = not suspicious, Red = copy-move evidence.
    """
    vis = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    color = (255, 50, 50) if is_forged else (50, 200, 50)

    for m in matches[:100]:  # Draw at most 100 lines to avoid clutter
        pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
        cv2.line(vis, pt1, pt2, color, 1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)

    return vis


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python copy_move.py <image_path> [ORB|SIFT]")
    else:
        method = sys.argv[2] if len(sys.argv) > 2 else "ORB"
        is_forged, n_matches, _ = detect_copy_move(
            sys.argv[1], method=method, save_path="copy_move_result.png"
        )
        print(f"\nResult: {'FORGED' if is_forged else 'AUTHENTIC'} ({n_matches} suspicious matches)")
