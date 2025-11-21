import cv2
import numpy as np
import pickle
import time
import os

# ---------------- Config ----------------
KNOWN_FILE = "known_faces.pkl"   # expects {"names": [...], "features": [[...], ...]}
FACE_SIZE = (80, 80)
THRESHOLD = 0.55       # smaller -> stricter matching; tune for your data
UNLOCK_DURATION = 4.0  # seconds to consider unlocked after a positive match
CHECK_INTERVAL = 0.05  # loop sleep interval (seconds)
# -------------------------------

# load Haar cascade
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# load known faces
if not os.path.exists(KNOWN_FILE):
    raise FileNotFoundError(f"Known faces file not found: {KNOWN_FILE}")

with open(KNOWN_FILE, "rb") as f:
    data = pickle.load(f)
    known_names = data.get("names", [])
    known_feats = [np.array(feat, dtype=np.float32) for feat in data.get("features", [])]

if not known_names or not known_feats:
    raise ValueError("No known faces or features found in the .pkl file.")

print(f"[INFO] Loaded {len(known_names)} known faces from {KNOWN_FILE}")

# ---------------- LBP histogram function ----------------
def compute_lbp_hist(gray):
    gray = cv2.resize(gray, FACE_SIZE)
    h, w = gray.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    neighbors = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, 0:-2], gray[1:-1, 0:-2]
    ]
    center = gray[1:-1, 1:-1]
    for i, nb in enumerate(neighbors):
        lbp |= ((nb >= center) << i).astype(np.uint8)
    hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 256))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

# ---------------- Unlock state ----------------
unlocked_until = 0.0
currently_unlocked = False

def trigger_unlock(name, dist):
    """Called when a positive match occurs."""
    global unlocked_until
    unlocked_until = max(unlocked_until, time.time() + UNLOCK_DURATION)
    print(f"UNLOCKED -> {name} (distance={dist:.3f}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    # UI overlay handled in main loop

# ---------------- Main loop ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Starting face unlock simulation. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        matched_name = None
        matched_dist = None

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            hist = compute_lbp_hist(face_gray)

            dists = [np.linalg.norm(hist - f) for f in known_feats]
            idx = int(np.argmin(dists))
            dist = float(dists[idx])

            # draw rectangle and distance
            cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 255, 180), 2)
            cv2.putText(frame, f"{dist:.3f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)

            if dist < THRESHOLD:
                matched_name = known_names[idx]
                matched_dist = dist
                # highlight match
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 3)
                cv2.putText(frame, f"{matched_name}", (x, y+h+24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                # unlock on first positive match
                trigger_unlock(matched_name, matched_dist)
                break  # stop after first good match

        # handle lock timer
        now = time.time()
        if now < unlocked_until:
            currently_unlocked = True
        else:
            currently_unlocked = False

        # overlay status
        status_text = "UNLOCKED" if currently_unlocked else "LOCKED"
        color = (0, 255, 0) if currently_unlocked else (0, 0, 255)
        cv2.putText(frame, f"STATUS: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if matched_name is not None:
            cv2.putText(frame, f"Match: {matched_name} ({matched_dist:.3f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Unlock (SIM)", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        time.sleep(CHECK_INTERVAL)

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting.")
