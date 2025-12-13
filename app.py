import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image
import av

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO("best.pt")

COIN_VALUES = {
    "half_pound": 0.5,
    "one_pound": 1.0
}

CONF_THRESHOLD = 0.6

# ----------------------------
# Helper Functions
# ----------------------------
def process_detections(results):
    """Process YOLO results and return counts and total money."""
    total_money = 0.0
    counts = {"half_pound": 0, "one_pound": 0}
    
    if results.boxes is not None and len(results.boxes) > 0:
        for cls, conf in zip(results.boxes.cls, results.boxes.conf):
            if conf < CONF_THRESHOLD:
                continue
            class_name = model.names[int(cls)]
            if class_name in counts:
                counts[class_name] += 1
                total_money += COIN_VALUES[class_name]
    
    return counts, total_money

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Egyptian Coin Detector", layout="centered")
st.title("🪙 Egyptian Coin Detection")
st.write("Detect **1 EGP** and **0.5 EGP** coins and calculate total money.")

# Create tabs for different detection modes
tab1, tab2 = st.tabs(["📸 Upload Image", "📱 Live Mobile Camera"])

# ----------------------------
# Tab 1: Image Upload
# ----------------------------
with tab1:
    st.header("📸 Upload an Image")
    st.write("Upload an image containing coins to detect and calculate the total amount.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        # Run inference (YOLO expects RGB numpy array)
        with st.spinner("Detecting coins..."):
            results = model(img_array, conf=CONF_THRESHOLD)[0]
        
        # Process detections
        counts, total_money = process_detections(results)
        
        # Draw detections manually on original RGB image to avoid color issues
        annotated_img = img_array.copy()
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                if conf < CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                class_name = model.names[int(cls)]
                
                # Draw bounding box (green color for RGB)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Convert to PIL Image (already RGB)
        annotated_img_pil = Image.fromarray(annotated_img)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Detected Coins")
            st.image(annotated_img_pil, caption="Detection Results", use_container_width=True)
        
        # Display statistics
        st.subheader("🧮 Detection Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("1 Pound Coins", counts['one_pound'])
        with col2:
            st.metric("0.5 Pound Coins", counts['half_pound'])
        with col3:
            st.metric("Total Amount", f"{total_money:.2f} EGP")
        
        # Example calculation display
        if total_money > 0:
            st.info(
                f"**Calculation:** "
                f"{counts['one_pound']} × 1 EGP + "
                f"{counts['half_pound']} × 0.5 EGP = "
                f"**{total_money:.2f} EGP**"
            )
        else:
            st.warning("No coins detected in the image. Please try another image.")

# ----------------------------
# Tab 2: Live Mobile Camera
# ----------------------------
with tab2:
    st.header("📱 Live Mobile Camera Detection")
    st.write(
        "Open this app from your **phone browser** using the **Network URL** "
        "and allow camera access to detect coins in real-time."
    )
    
    # Video Processor Class
    class CoinDetector(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Run inference
            results = model(img, conf=CONF_THRESHOLD)[0]
            
            # Process detections
            counts, total_money = process_detections(results)
            
            # Draw detections
            annotated = results.plot()
            
            # Overlay total money on frame
            cv2.putText(
                annotated,
                f"Total: {total_money:.2f} EGP",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Overlay coin counts
            cv2.putText(
                annotated,
                f"1 EGP: {counts['one_pound']} | 0.5 EGP: {counts['half_pound']}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
    
    # WebRTC Streamer
    webrtc_streamer(
        key="coin-detection",
        video_processor_factory=CoinDetector,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
