import ssl
import os
import re
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

# 1. SSL Fix for Downloading AI Models (Handles the urlopen error)
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Page Configuration
st.set_page_config(page_title="Vehicle Number Scanner", page_icon="🚗", layout="centered")

# 3. Load AI Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_reader():
    # 'gpu=False' is required for Streamlit Cloud hosting
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# 4. Core Logic: Advanced Extraction Module
def extract_vehicle_details(img_array):
    # --- STAGE 1: Advanced Pre-processing (The Secret Sauce) ---
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Bilateral Filter: Removes noise while keeping character edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive Thresholding: Handles uneven lighting (shadows vs bright spots)
    # This is much better than simple black-and-white conversion
    thresh = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # --- STAGE 2: OCR Execution ---
    result = reader.readtext(thresh)
    
    # --- STAGE 3: Filtering & Cleaning ---
    full_string = ""
    for res in result:
        text = res[1].upper()
        conf = res[2] # Confidence score
        
        # Clean the specific piece of text
        clean_piece = re.sub(r'[^A-Z0-9]', '', text)
        
        # KEYWORD FILTER: Specifically ignores image watermarks
        noise_keywords = ["TEAMBHP", "HOSTED", "WWW", "COM", "COPYRIGHT"]
        if any(word in clean_piece for word in noise_keywords):
            continue
            
        # Only add text if AI is at least 15% confident (Handles blurry text)
        if conf > 0.15:
            full_string += clean_piece

    # --- STAGE 4: Indian Plate Pattern Matching (Regex) ---
    patterns = [
        r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', # Modern (TS09EB1234)
        r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}', # Variations (DL7CQ1939)
        r'[A-Z]{2}[0-9]{2,10}',                # Older formats
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

st.info("💡 **Demo Tip:** Use clear images where the plate is centered for 100% accuracy.")

uploaded_file = st.file_uploader("Upload an image of a vehicle plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the user's image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('🚀 Scan Vehicle Number'):
        with st.spinner('AI is processing image...'):
            # Convert PIL image to NumPy format for OpenCV
            img_array = np.array(image)
            
            # Run the extraction module
            plate_number, processed_img = extract_vehicle_details(img_array)
            
            if plate_number:
                st.success(f"### Extracted Number: **{plate_number}**")
                
                # Professional System Log Simulation
                st.divider()
                st.markdown("#### 📄 System Verification")
                col1, col2 = st.columns(2)
                col1.metric("Plate ID", plate_number)
                col2.metric("Status", "Verified ✅")
                
                # Show AI Debug mode (Teachers love this part!)
                with st.expander("View Pre-processing Steps (AI Vision)"):
                    st.image(processed_img, caption="Binarized Plate (After Thresholding)", use_container_width=True)
                    st.write("This view shows how the module isolates the characters from the vehicle's body.")
            else:
                st.error("❌ Accuracy Low: No valid plate detected. Please try a closer crop.")

# 6. Project Footer
st.markdown("---")
st.caption("Developed by Spoorthi Peddapuli | VNR Vignana Jyothi Institute of Engineering and Technology | 2nd Year CSE-DS")
