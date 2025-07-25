import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO("best.pt")  # Replace with your model path

st.set_page_config(page_title="Helmet Detection App", layout="wide")
st.title("🪖 Helmet Detection using YOLO")

# Function to perform detection
def detect_frame(frame):
    results = model(frame, imgsz=640, conf=0.5)
    annotated_frame = results[0].plot()
    return annotated_frame

# Sidebar for input type
input_type = st.sidebar.selectbox("Choose input type", ["Image", "Video", "Webcam"])

# ===================== IMAGE MODE =====================
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting..."):
            result_img = detect_frame(np.array(image))
            st.image(result_img, caption="Detected Output", use_column_width=True)

# ===================== VIDEO MODE =====================
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.info("Press stop to end video processing")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_frame(frame)
            stframe.image(detected_frame, channels="RGB")

        cap.release()

# ===================== WEBCAM MODE =====================
elif input_type == "Webcam":
    st.info("Turn on your webcam below")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_frame(frame)
            stframe.image(detected_frame, channels="RGB")

        cap.release()
