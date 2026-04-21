# inst/python/align.py
"""Image alignment using SIFT features."""

import cv2
import numpy as np


def align_images(ppl_path, xpl_path, output_path, method='similarity'):
    """
    Align PPL image to XPL reference using SIFT features.
    
    Args:
        ppl_path: Path to plane-polarized light image
        xpl_path: Path to cross-polarized light image (reference)
        output_path: Path to save aligned PPL image
        method: 'similarity' (with fallback) or 'translation' (force translation only)
    
    Returns:
        dict with n_matches and method_used
    """
    ppl = cv2.imread(ppl_path, cv2.IMREAD_COLOR)
    xpl = cv2.imread(xpl_path, cv2.IMREAD_COLOR)
    
    if ppl is None or xpl is None:
        raise ValueError("Could not read images")
    
    ppl_gray = cv2.cvtColor(ppl, cv2.COLOR_BGR2GRAY)
    xpl_gray = cv2.cvtColor(xpl, cv2.COLOR_BGR2GRAY)
    
    # SIFT with conservative parameters
    sift = cv2.SIFT_create(
        nfeatures=2000,
        contrastThreshold=0.04,
        edgeThreshold=10
    )
    kp_ppl, des_ppl = sift.detectAndCompute(ppl_gray, None)
    kp_xpl, des_xpl = sift.detectAndCompute(xpl_gray, None)
    
    if des_ppl is None or des_xpl is None:
        raise ValueError("No features detected")
    
    # FLANN matcher
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )
    matches = matcher.knnMatch(des_ppl, des_xpl, k=2)
    
    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.65 * n.distance]
    
    if len(good) < 4:
        raise ValueError(f"Only {len(good)} good matches found (need ≥4)")
    
    src_pts = np.float32([kp_ppl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_xpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    method_used = method
    
    if method == 'similarity':
        # Try similarity transform with sanity checks
        try:
            M, _ = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if M is not None:
                # Verify transform is reasonable for remounted slide
                scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                rotation_deg = np.abs(np.arctan2(M[0, 1], M[0, 0])) * 180 / np.pi
                
                if scale < 0.9 or scale > 1.1:
                    raise ValueError(f"Scale {scale:.3f} out of bounds [0.9, 1.1]")
                if rotation_deg > 5:
                    raise ValueError(f"Rotation {rotation_deg:.1f}° too large (max 5°)")
                
                H = np.vstack([M, [0, 0, 1]])
            else:
                raise ValueError("Similarity transform estimation failed")
                
        except Exception as e:
            # Fallback to translation only (more robust for noisy matches)
            print(f"Similarity transform failed ({e}), falling back to translation")
            method_used = 'translation_fallback'
            dx = np.median(dst_pts[:, 0, 0] - src_pts[:, 0, 0])
            dy = np.median(dst_pts[:, 0, 1] - src_pts[:, 0, 1])
            H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    else:
        # Translation only
        dx = np.median(dst_pts[:, 0, 0] - src_pts[:, 0, 0])
        dy = np.median(dst_pts[:, 0, 1] - src_pts[:, 0, 1])
        H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
    
    h, w = xpl.shape[:2]
    aligned = cv2.warpPerspective(ppl, H, (w, h), flags=cv2.INTER_LINEAR)
    
    cv2.imwrite(output_path, aligned)
    
    return {
        'n_matches': len(good),
        'method_used': method_used
    }
