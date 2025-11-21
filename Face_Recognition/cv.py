import cv2
import numpy as np
import pickle
from datetime import datetime

# ---------------- Configuration ----------------
KNOWN_FILE = "known_faces.pkl"
FACE_SIZE = (80, 80)
THRESHOLD = 0.55
TOP_FEATURES = 5
POINTER_FONT_SCALE = 0.4
# ------------------------------------------------

haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- Load Known Faces ----------------
try:
    with open(KNOWN_FILE, "rb") as f:
        data = pickle.load(f)
        known_names = data["names"]
        known_feats = [np.array(f, dtype=np.float16) for f in data["features"]]
    print(f"Loaded {len(known_names)} known faces.")
except FileNotFoundError:
    known_names, known_feats = [], []
    print("No existing data found.")

# ---------------- LBP Feature Extraction ----------------
def compute_lbp_fast(gray):
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
    hist /= (hist.sum() + 1e-7)
    return lbp, hist.astype(np.float16)

# ---------------- Step 1: Registration ----------------
user_name = input("Enter the name of the person to register: ").strip()
if user_name == "":
    user_name = "User_" + datetime.now().strftime("%H%M%S")
print(f"Starting registration for: {user_name}")

cap = cv2.VideoCapture(0)
registered = False

while not registered:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        _, hist = compute_lbp_fast(face_gray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Press 's' to save", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('s') and len(faces) > 0:
        known_names.append(user_name)
        known_feats.append(hist)
        with open(KNOWN_FILE, "wb") as f:
            pickle.dump({"names": known_names, "features": [f.tolist() for f in known_feats]}, f)
        print(f"✅ Registered {user_name}")
        registered = True
    elif key == ord('q'):
        print("Registration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ---------------- Step 2: Live Detection ----------------
print("Starting live face detection. Press 'q' to quit.")
cap = cv2.VideoCapture(0)
extracted_features = []

descs = [
    "Texture smoothness", "Edge sharpness", "Cheek brightness",
    "Skin uniformity", "Facial symmetry", "Shadow contrast"
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    # Sort faces left-to-right to assign consistent tags
    faces = sorted(faces, key=lambda x: x[0])

    for i, (x, y, w, h) in enumerate(faces):
        face_gray = gray[y:y+h, x:x+w]
        lbp, hist = compute_lbp_fast(face_gray)

        name = "Unknown"
        min_dist = 1e9

        if known_feats:
            dists = [np.linalg.norm(hist - f) for f in known_feats]
            idx = int(np.argmin(dists))
            min_dist = dists[idx]
            if min_dist < THRESHOLD:
                name = known_names[idx]

        # Label face with ID number
        label = f"{name} (Face #{i+1})"
        color = (0, 255, 100) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Top unique features
        top_bins = np.argsort(hist)[-TOP_FEATURES:]
        h_lbp, w_lbp = lbp.shape
        step_h, step_w = h_lbp // 8, w_lbp // 8

        for j, ub in enumerate(top_bins):
            row = (ub // 8) * step_h + step_h // 2
            col = (ub % 8) * step_w + step_w // 2
            row_scaled = y + int(row * (h / h_lbp))
            col_scaled = x + int(col * (w / w_lbp))
            tail_x = col_scaled + 40
            tail_y = row_scaled - 40
            cv2.arrowedLine(frame, (tail_x, tail_y), (col_scaled, row_scaled),
                            (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(frame, descs[j % len(descs)], (tail_x - 50, tail_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, POINTER_FONT_SCALE, (0, 200, 255), 1)

        extracted_features.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "face_id": i + 1,
            "name": name,
            "distance": float(min_dist),
            "top_features": [(int(b), float(hist[b])) for b in top_bins]
        })

    cv2.imshow("Multi-Face Detection", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- Session Summary ----------------
print("\n===== Session Feature Summary =====")
for i, f in enumerate(extracted_features[-5:]):
    print(f"[{i+1}] {f['time']} | Face#{f['face_id']} | {f['name']} | Dist={f['distance']:.3f} | Top bins: {f['top_features']}")
print("===================================")
