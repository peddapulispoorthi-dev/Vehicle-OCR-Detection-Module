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
    # 1. Preprocessing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 2. OCR 
    result = reader.readtext(thresh)
    
    # 3. Smart Filtering
    all_detected_text = ""
    candidates = []

    for res in result:
        text = res[1].upper()
        conf = res[2]
        
        # Clean text of symbols for checking
        clean_subtext = re.sub(r'[^A-Z0-9]', '', text)
        
        # Only ignore text if it's very low confidence or likely a watermark
        if conf > 0.15 and "TEAMBHP" not in clean_subtext and "HOSTED" not in clean_subtext:
            candidates.append(clean_subtext)
    
    full_string = "".join(candidates)
    
    # 4. Refined Indian Plate Patterns
    # We use a more flexible regex to catch plates even if there are small gaps
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}', # Standard: WB06F9209
        r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}', # DL7CQ1939
        r'[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}[0-9]{0,4}' # Fallback for old/partial plates
    ]
    
    for p in patterns:
        match = re.search(p, full_string)
        if match:
            return match.group(), thresh

    return full_string[:10] if full_string else None, thresh

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
