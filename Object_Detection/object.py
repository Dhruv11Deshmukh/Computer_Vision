import cv2
import numpy as np

# ---------- PARAMETERS (tweak if needed) ----------
# aspect ratio = long_side / short_side
PEN_ASPECT_MIN = 6.0    # pens are relatively shorter/thicker (lower ratio)
PEN_ASPECT_MAX = 18.0
PENCIL_ASPECT_MIN = 18.0
PENCIL_ASPECT_MAX = 60.0

MIN_CONTOUR_AREA = 800  # ignore smaller contours (tweak with image size)
SOLIDITY_MIN = 0.25     # area / rect_area (filter very hollow shapes)

KMEANS_K = 2            # for dominant color extraction from ROI

# ---------- HELPER FUNCTIONS ----------
def dominant_hsv_color(roi_bgr):
    """Return dominant HSV color (h, s, v) using cv2.kmeans on the ROI pixels."""
    if roi_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3)).astype(np.float32)
    # remove near-black/near-white pixels to focus on object color
    lower_val_mask = (pixels[:, 2] > 30) & (pixels[:, 2] < 245)
    if lower_val_mask.sum() < 10:
        pixels = pixels
    else:
        pixels = pixels[lower_val_mask]

    if len(pixels) < 10:
        # fallback to mean
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        return [int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, KMEANS_K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]
    return [int(dominant[0]), int(dominant[1]), int(dominant[2])]


def extract_rotated_roi(img, rect):
    """Extract rotated rectangle ROI from image.
    rect = ((cx,cy),(w,h), angle) from cv2.minAreaRect
    returns the ROI image (BGR)
    """
    (cx, cy), (w, h), angle = rect
    if w <= 0 or h <= 0:
        return None
    # get rotation matrix for the rectangle
    # we want to rotate the whole image so the rect becomes axis aligned
    center = (int(cx), int(cy))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h_img, w_img = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
    # compute the size of the upright rect and crop
    size_w, size_h = int(w), int(h)
    # after rotation the rect angle is 0; but need new center location
    new_cx, new_cy = int(center[0]), int(center[1])
    x1 = max(0, new_cx - size_w // 2)
    y1 = max(0, new_cy - size_h // 2)
    x2 = min(w_img, x1 + size_w)
    y2 = min(h_img, y1 + size_h)
    roi = rotated[y1:y2, x1:x2]
    return roi


def draw_rotated_box(img, rect, label, color=(0,255,0), thickness=2):
    """Draw the rotated rectangle box and put a label near it."""
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(img, [box], 0, color, thickness)
    # place label at top-left of box
    xs = box[:,0]
    ys = box[:,1]
    tl = (int(xs.min()), int(ys.min()) - 10)
    cv2.putText(img, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def classify_by_aspect_and_color(long_side, short_side, hsv_dom):
    """Rule-based classification combining aspect ratio and color hints."""
    if short_side == 0:
        return "Unknown"
    aspect = float(long_side) / float(short_side)
    # strong aspect-based decisions
    if PEN_ASPECT_MIN <= aspect <= PEN_ASPECT_MAX:
        # pen-like
        return "Pen"
    if PENCIL_ASPECT_MIN <= aspect <= PENCIL_ASPECT_MAX:
        return "Pencil"

    # if ambiguous, use color heuristics (hue)
    if hsv_dom is not None:
        h, s, v = hsv_dom
        # pencil wood/yellow typically hue roughly 10-40 and medium-high value
        if 10 <= h <= 45 and v > 60:
            return "Pencil"
        # pens often are blue/black/dark or colored plastic with higher saturation
        if (h < 10 or h > 150) or (s > 60 and v > 30):
            return "Pen"
    return "Unknown"

# ---------- MAIN FLOW ----------
def analyze_image(path_to_image, save_out="output_oriented.jpg", visualize=True):
    img = cv2.imread(path_to_image)
    if img is None:
        print("Error: couldn't load image:", path_to_image)
        return

    orig = img.copy()
    # Resize to a working size if very large (optional)
    max_dim = 1280
    h0, w0 = img.shape[:2]
    if max(h0, w0) > max_dim:
        scale = max_dim / float(max(h0, w0))
        img = cv2.resize(img, (int(w0*scale), int(h0*scale)))
    proc = img.copy()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # adaptive threshold followed by closing to handle uneven lighting
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    counts = {"Pen": 0, "Pencil": 0, "Unknown": 0}
    annotated = proc.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # min area rect gives rotated bounding box: ((cx,cy),(w,h),angle)
        rect = cv2.minAreaRect(cnt)
        (cx,cy), (w,h), angle = rect
        if w <= 0 or h <= 0:
            continue

        long_side = max(w,h)
        short_side = min(w,h)
        rect_area = w * h
        solidity = area / rect_area if rect_area > 0 else 0

        # filter very non-solid blobs
        if solidity < SOLIDITY_MIN:
            # could be a cluster of touching items or noise - still try
            pass

        # get rotated ROI to examine color/texture
        roi = extract_rotated_roi(proc, rect)
        hsv_dom = dominant_hsv_color(roi) if roi is not None else None

        label = classify_by_aspect_and_color(long_side, short_side, hsv_dom)

        # draw rotated box and label on annotated image
        color = (0,255,0) if label == "Pen" else ((255,0,0) if label=="Pencil" else (128,128,128))
        draw_rotated_box(annotated, rect, f"{label}", color=color, thickness=2)

        counts[label] = counts.get(label, 0) + 1

    # summary text
    summary = f"Pens: {counts.get('Pen',0)}   Pencils: {counts.get('Pencil',0)}   Unknown: {counts.get('Unknown',0)}"
    cv2.rectangle(annotated, (5,5), (600,45), (0,0,0), -1)
    cv2.putText(annotated, summary, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imwrite(save_out, annotated)
    if visualize:
        cv2.imshow("Analysis (orientation invariant)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Saved result to:", save_out)
    print(summary)
    return annotated

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect pens and pencils in an image (orientation-invariant).")
    parser.add_argument("image", help="Path to input image (cluster image)")
    parser.add_argument("--out", help="Output image path", default="output_oriented.jpg")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not display result window")
    args = parser.parse_args()
    analyze_image(args.image, save_out=args.out, visualize=args.show)
