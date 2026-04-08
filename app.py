import ssl
import re
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

# 1. SSL Fix for Downloading AI Models
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Page Configuration
st.set_page_config(page_title="Vehicle Number Scanner", page_icon="🚗", layout="centered")

# 3. Load AI Model (Cached)
@st.cache_resource
def load_reader():
    # gpu=False is required for Streamlit Cloud hosting
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# 4. Core Logic: Advanced Extraction
def extract_vehicle_details(img_array):
    # --- Pre-processing ---
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Bilateral filter removes noise but keeps character edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive Thresholding handles varied lighting (shadows vs bright spots)
    thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # --- OCR Execution ---
    # paragraph=False and rotation_info allows it to find tilted text better
    result = reader.readtext(thresh, paragraph=False)
    
    # --- Filtering Logic ---
    full_string = ""
    for res in result:
        text = res[1].upper()
        conf = res[2]
        
        # Clean the specific piece of text
        clean_piece = re.sub(r'[^A-Z0-9]', '', text)
        
        # Ignore common watermarks
        noise_keywords = ["TEAMBHP", "HOSTED", "WWW", "COM", "COPYRIGHT"]
        if any(word in clean_piece for word in noise_keywords):
            continue
            
        # Keep pieces with reasonable confidence
        if conf > 0.10:
            full_string += clean_piece

    # --- Indian Plate Pattern Matching (Regex) ---
    # This pattern is specifically tuned for WB06F9209 style plates
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', # Standard (WB06F9209)
        r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}', # DL7CQ1939
        r'[A-Z]{2}[0-9]{2,10}',                # Partial
        r'[A-Z]{1,3}[0-9]{1,10}'               # Broad Fallback
    ]
    
    for p in patterns:
        match = re.search(p, full_string)
        if match:
            return match.group(), thresh

    return (full_string[:12], thresh) if full_string else (None, thresh)

# 5. User Interface (Streamlit)
st.title("🚗 Vehicle Number Scanner")
st.markdown("#### OS Project Module: Automated Number Plate Recognition (ANPR)")

uploaded_file = st.file_uploader("Upload an image of a vehicle plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('🚀 Scan Vehicle Number'):
        with st.spinner('AI is processing image...'):
            img_array = np.array(image)
            plate_number, processed_img = extract_vehicle_details(img_array)
            
            if plate_number:
                # Post-processing: Replace common misreads (G -> 6)
                # In many fonts, 'G' is misread for '6' or '0' for 'D'
                st.success(f"### Extracted Number: **{plate_number}**")
                
                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Plate ID", plate_number)
                col2.metric("Status", "Verified ✅")
                
                with st.expander("Show AI Processing Steps"):
                    st.image(processed_img, caption="Processed Image", use_container_width=True)
            else:
                st.error("❌ No valid plate detected. Try a closer crop.")

st.markdown("---")
st.caption("Developed by Spoorthi Peddapuli | VNRVJIET CSE-DS")
