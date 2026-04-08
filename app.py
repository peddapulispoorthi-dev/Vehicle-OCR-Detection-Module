import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Vehicle Scanner", page_icon="🚗", layout="centered")

# 2. Load the AI Model (Cached so it stays fast)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# 3. Core Logic Function
def process_license_plate(img_array):
    height, width = img_array.shape[:2]
    
    # 1. Preprocessing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 2. OCR with Spatial Filtering
    result = reader.readtext(thresh)
    
    valid_parts = []
    for res in result:
        coords = res[0] # List of 4 corners: [[x,y], [x,y], [x,y], [x,y]]
        text = res[1].upper()
        conf = res[2]
        
        # Calculate the vertical center of this specific text box
        text_y_center = (coords[0][1] + coords[2][1]) / 2
        
        # SPATIAL FILTER: Ignore text in the top 20% or bottom 20% of the image
        # This kills watermarks like "Team-BHP" or "Hosted on"
        if 0.20 * height < text_y_center < 0.80 * height:
            if conf > 0.25:
                valid_parts.append(text)
    
    clean_text = re.sub(r'[^A-Z0-9]', '', "".join(valid_parts))
    
    # 3. Indian Plate Pattern Match
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', # WB06F9209
        r'[A-Z]{2}[0-9]{1,6}[A-Z0-9]{0,4}'      # Fallback for older plates
    ]
    
    for p in patterns:
        match = re.search(p, clean_text)
        if match:
            return match.group(), thresh

    return clean_text, thresh

# 4. User Interface (UI)
st.title("🚗 Vehicle Number Plate Scanner")
st.markdown("### OS Project Module: Automated OCR Extraction")
st.info("Note: For best results, ensure the number plate is clearly visible in the crop.")

uploaded_file = st.file_uploader("Upload Plate Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('🚀 Extract Details'):
        with st.spinner('AI is processing image...'):
            img_array = np.array(image)
            plate_no, processed_img = process_license_plate(img_array)
            
            if plate_no:
                st.success(f"### Extracted Number: **{plate_no}**")
                
                # Mock Database Response
                with st.expander("View System Logs"):
                    st.json({
                        "Vehicle_No": plate_no,
                        "Status": "Verified",
                        "Location_Log": "VNRVJIET_GATE_1",
                        "Confidence": "High"
                    })
                
                # Show the 'Computer Vision' view for the viva
                if st.checkbox("Show Pre-processed Image (What the AI sees)"):
                    st.image(processed_img, caption="Thresholded Image", cmap='gray')
            else:
                st.error("No valid number plate pattern detected. Please try a closer crop.")

# 5. Footer
st.markdown("---")
st.caption("Developed by Spoorthi Peddapuli | VNRVJIET CSE-DS | OS Project 2026")
