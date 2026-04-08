import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

# Initialize the OCR reader
@st.cache_resource
def load_reader():
    # 'gpu=False' is used for CPU-based cloud hosting; set to True if running locally with NVIDIA
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

def extract_number(img_array):
    # 1. Preprocessing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Bilateral filter removes noise but keeps edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Thresholding for better contrast
    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. OCR with Confidence Scores
    result = reader.readtext(thresh)
    
    # 3. Filtering and Cleaning
    raw_parts = []
    for res in result:
        text = res[1]
        confidence = res[2]
        
        # Filtering: Only keep text if AI is more than 35% sure
        # This helps ignore tiny background text/watermarks
        if confidence > 0.35:
            raw_parts.append(text.upper())
    
    full_string = "".join(raw_parts)
    
    # Clean the text (Remove symbols and spaces)
    clean_text = re.sub(r'[^A-Z0-9]', '', full_string)
    
    # 4. Indian License Plate Logic (Regex)
    # This specifically looks for State Code (2 letters) + District (2 numbers)...
    # This helps ignore things like "TEAM" at the end or "IND" at the start.
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', # Modern: DL7CQ1939
        r'[A-Z]{2}[0-9]{2}[0-9]{4}',           # Older: AP091234
        r'[A-Z]{3}[0-9]{4}'                    # Very Old: ABC1234
    ]
    
    for p in patterns:
        match = re.search(p, clean_text)
        if match:
            return match.group()

    return clean_text

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Vehicle Scanner", page_icon="🚗", layout="centered")

st.title("🚗 Vehicle Number Plate Scanner")
st.markdown("### OS Project: Number Plate Extraction Module")
st.write("Upload a cropped image to extract the registration number.")

uploaded_file = st.file_uploader("Choose a plate image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Target Image', use_container_width=True)
    
    if st.button('🚀 Extract Vehicle Number'):
        with st.spinner('Analyzing characters...'):
            # Convert to OpenCV format
            img_array = np.array(image)
            
            # Process
            plate_number = extract_number(img_array)
            
            if plate_number:
                st.success(f"✅ Extracted Number: **{plate_number}**")
                
                # Visual Database Lookup Simulation
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vehicle ID", plate_number)
                with col2:
                    st.metric("System Status", "Verified")
            else:
                st.error("❌ Accuracy low. Please try a closer, clearer crop of the plate.")

st.markdown("---")
st.caption("Developed by Spoorthi Peddapuli | VNRVJIET CSE-DS")
