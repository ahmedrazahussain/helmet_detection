import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import os

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path

st.set_page_config(page_title="Helmet Detection", layout="wide")
st.title("🪖 Helmet Detection using YOLOv8")

# YOLO prediction function
def detect_image(img):
    results = model(img, conf=0.5)
    return results[0].plot()

# Sidebar for input choice
input_option = st.sidebar.radio("Select Input Type", ["Image", "Video", "Webcam"])

# ======================================
# IMAGE MODE
# ======================================
if input_option == "Image":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting..."):
                result_img = detect_image(img_np)
                st.image(result_img, caption="Detected Image", use_column_width=True)

                # Save result
                output_path = "output_image.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                with open(output_path, "rb") as file:
                    st.download_button("Download Detected Image", file, "helmet_detected.jpg")

# ======================================
# VIDEO MODE
# ======================================
elif input_option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("Run Detection on Video"):
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            out_path = "output_video.mp4"
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=0.5)
                annotated = results[0].plot()
                out.write(annotated)
                stframe.image(annotated, channels="RGB")

            cap.release()
            out.release()

            with open(out_path, "rb") as file:
                st.download_button("Download Processed Video", file, "helmet_detected_video.mp4")

# ======================================
# WEBCAM MODE
# ======================================
elif input_option == "Webcam":
    st.info("Allow browser webcam access below")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=0.3)
            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="helmet-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
