
import cv2
import face_recognition
import numpy as np
import os
import pickle

DB_FILE = "face_encodings.pkl"

def load_encodings():
    """Load encodings database from file if exists"""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_encodings(encodings):
    """Save encodings database"""
    with open(DB_FILE, "wb") as f:
        pickle.dump(encodings, f)


def enroll_face(frame, name):
    """Enroll new face into database"""
    enc_db = load_encodings()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, faces)
    if encs:
        enc_db[name] = encs[0]
        save_encodings(enc_db)
        return True
    return False


def find_known_face(frame):
    """Returns (names, faces) always — even if no encodings exist"""
    enc_db = load_encodings()
    if not enc_db:
        return [], []  

    known_names = list(enc_db.keys())
    known_encs = list(enc_db.values())
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, faces)

    names = []
    for e in encs:
        results = face_recognition.compare_faces(known_encs, e, tolerance=0.5)
        if True in results:
            idx = results.index(True)
            names.append(known_names[idx])
        else:
            names.append(None)
    return names, faces


def analyze_face(frame):
    """Analyze detected face(s) and return details"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, faces)

    analysis_results = []
    for (top, right, bottom, left), enc in zip(faces, encs):
        face_width = right - left
        face_height = bottom - top
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        details = {
            "face_width": face_width,
            "face_height": face_height,
            "center": (center_x, center_y),
        }
        landmarks = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
        if landmarks:
            lm = landmarks[0]
            if "left_eye" in lm and "right_eye" in lm:
                left_eye = np.mean(lm["left_eye"], axis=0)
                right_eye = np.mean(lm["right_eye"], axis=0)
                eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
                details["eye_distance"] = int(eye_distance)
        details["emotion"] = "neutral"
        analysis_results.append(details)
    return analysis_results

def capture_and_analyze():
    """Open webcam, capture face, analyze & return result"""
    cap = cv2.VideoCapture(0)
    print("Showing camera. Press 'q' to capture & quit.")

    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame is None:
        return None
    names, faces = find_known_face(frame)
    analysis = analyze_face(frame)

    return {"names": names, "faces": faces, "analysis": analysis}


