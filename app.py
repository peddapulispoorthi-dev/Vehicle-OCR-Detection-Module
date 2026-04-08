import ssl
import os
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

# 4. Core Logic Function
def extract_vehicle_details(img_array):
    # --- Pre-processing ---
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Bilateral filter removes noise but keeps edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive Thresholding handles varied lighting conditions
    thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # --- OCR Execution ---
    result = reader.readtext(thresh)
    
    # --- Filtering Logic ---
    full_string = ""
    for res in result:
        text = res[1].upper()
        conf = res[2]
        
        # Clean the specific piece of text
        clean_piece = re.sub(r'[^A-Z0-9]', '', text)
        
        # KEYWORD FILTER: Ignore common image watermarks
        noise_keywords = ["TEAMBHP", "HOSTED", "WWW", "COM", "COPYRIGHT"]
        if any(word in clean_piece for word in noise_keywords):
            continue
            
        # Only add text if AI is at least 15% confident
        if conf > 0.15:
            full_string += clean_piece

    # --- Indian Plate Pattern Matching (Regex) ---
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', # Standard (WB06F9209)
        r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}', # Delhi/Small states (DL7CQ1939)
        r'[A-Z]{2}[0-9]{1,10}',                # Partial or old plates
        r'[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}[0-9]{0,4}' # Extreme fallback
    ]
    
    for p in patterns:
        match = re.search(p, full_string)
        if match:
            return match.group(), thresh

    # Final fallback: return the raw cleaned string if no pattern matches
    return (full_string[:12], thresh) if full_string else (None, thresh)

# 5. User Interface (Streamlit)
st.title("🚗 Vehicle Number Scanner")
st.markdown("#### OS Project Module: Automated Number Plate Recognition (ANPR)")

st.info("💡 **Pro-Tip:** For the best accuracy, use a clear, closely cropped image of the number plate.")

uploaded_file = st.file_uploader("Upload an image of a vehicle plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('🚀 Scan Vehicle Number'):
        with st.spinner('AI is analyzing the plate...'):
            # Convert PIL to NumPy/OpenCV format
            img_array = np.array(image)
            
            # Process the image
            plate_number, processed_img = extract_vehicle_details(img_array)
            
            if plate_number:
                st.success(f"### Extracted Number: **{plate_number}**")
                
                # Visual Database Simulation
                st.divider()
                st.markdown("#### 📄 System Logs")
                col1, col2, col3 = st.columns(3)
                col1.metric("Plate ID", plate_number)
                col2.metric("Status", "Authenticated")
                col3.metric("Database", "VNR-VJIET-OS")
                
                # Show AI vision debug mode
                with st.expander("Show AI Vision (Pre-processed Image)"):
                    st.image(processed_img, caption="What the AI sees (Binarized)", use_container_width=True)
                    st.caption("This image shows how we use Thresholding to isolate characters from the background.")
            else:
                st.error
