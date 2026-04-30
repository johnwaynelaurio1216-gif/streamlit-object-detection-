import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os

# ---------- UI STYLE (MINIMALIST) ----------
st.set_page_config(page_title="AI Detection", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    text-align: center;
    font-weight: 600;
}
.stSidebar {
    background-color: #111827;
}
.css-1d391kg {
    background-color: rgba(255,255,255,0.05);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- HEADER ----------
st.markdown("<h1>🎥 AI Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Minimal • Real-time • Smart Tracking</p>", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)
target_object = st.sidebar.text_input("Alert Object", "cell phone")
save_frames = st.sidebar.checkbox("Save Frames")

# VIDEO EFFECTS
st.sidebar.subheader("🎨 Effects")
grayscale = st.sidebar.checkbox("Grayscale")
blur = st.sidebar.checkbox("Blur")

# Create folder
if save_frames and not os.path.exists("captures"):
    os.makedirs("captures")

# FPS tracker
prev_time = 0

# ---------- VIDEO CALLBACK ----------
def video_frame_callback(frame):
    global prev_time

    img = frame.to_ndarray(format="bgr24")

    # YOLO Detection
    results = model.track(img, persist=True, conf=confidence, verbose=False)
    annotated_frame = results[0].plot()

    names = model.names
    detected_objects = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            detected_objects.append(names[cls_id])

    object_count = len(detected_objects)

    # ---------- ALERT ----------
    if target_object.lower() in detected_objects:
        cv2.rectangle(annotated_frame, (10, 10), (400, 70), (0, 0, 255), -1)
        cv2.putText(
            annotated_frame,
            f"ALERT: {target_object}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # ---------- COUNT ----------
    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # ---------- FPS ----------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ---------- VIDEO EFFECTS ----------
    if grayscale:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)

    if blur:
        annotated_frame = cv2.GaussianBlur(annotated_frame, (15, 15), 0)

    # ---------- SAVE ----------
    if save_frames and object_count > 0:
        filename = f"captures/frame_{int(time.time())}.jpg"
        cv2.imwrite(filename, annotated_frame)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ---------- STREAM ----------
webrtc_streamer(
    key="minimal-ai-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)