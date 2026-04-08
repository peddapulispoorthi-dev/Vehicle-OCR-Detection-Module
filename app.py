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
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Noise Reduction: Bilateral Filter preserves edges better than Gaussian blur
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive Thresholding: Crucial for "working for all" lighting conditions
    thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Run OCR
    result = reader.readtext(thresh)
    
    # Filter by confidence and clean text
    raw_text = ""
    for res in result:
        if res[2] > 0.20:  # Lowered threshold to pick up faint numbers
            raw_text += res[1].upper()
    
    # Remove all non-alphanumeric characters
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)
    
    # Pattern matching for Indian License Plates
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', # Modern (TS09EB1234)
        r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}', # Variations (DL7CQ1939)
        r'[A-Z]{2}[0-9]{6}',                   # Older formats
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
