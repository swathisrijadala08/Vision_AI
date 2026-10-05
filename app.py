import os
import urllib.request

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Vision AI", page_icon="👁️", layout="wide")

st.title("👁️ Vision AI")
st.write("Hand Detection • Finger Counting • Gesture Recognition")

# ---------------- MODEL SETUP ----------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@st.cache_resource
def load_landmarker():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading hand model (one time only)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# ---------------- FUNCTIONS ----------------
def count_fingers(lm):
    """Count raised fingers from 21 hand landmarks."""
    fingers = []

    # Thumb: tip is farther from the pinky base than the thumb joint
    def dist(a, b):
        return np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    fingers.append(1 if dist(4, 17) > dist(3, 17) else 0)

    # Index, Middle, Ring, Pinky
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)

    return sum(fingers)


def get_gesture(count):
    return {
        0: "✊ Fist",
        1: "☝️ One",
        2: "✌️ Two",
        3: "🤟 Three",
        4: "🖖 Four",
        5: "🖐️ Open Hand",
    }.get(count, "Unknown")


def draw_landmarks(image, lm):
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)


landmarker = load_landmarker()

# ---------------- CAMERA ----------------
picture = st.camera_input("📷 Show your hand to the camera")

if picture:
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        st.success(f"✅ {len(results.hand_landmarks)} hand(s) detected!")

        total_fingers = 0

        for i, lm in enumerate(results.hand_landmarks):
            side = results.handedness[i][0].category_name
            finger_count = count_fingers(lm)
            total_fingers += finger_count

            draw_landmarks(image, lm)

            st.subheader(f"Hand {i + 1}")
            st.write(f"**Side:** {side}")
            st.write(f"**Fingers:** {finger_count}")
            st.write(f"**Gesture:** {get_gesture(finger_count)}")

        st.metric("Total Fingers Detected", total_fingers)