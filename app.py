import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ... the rest of your imports start here ...
import streamlit as st
import cv2
import easyocr
# ... rest of the code
import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

# Initialize the OCR reader
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

def extract_number(img_array):
    # Preprocessing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR
    result = reader.readtext(thresh)
    raw_text = "".join([res[1] for res in result])
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return clean_text

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Vehicle Scanner", page_icon="🚗")

st.title("🚗 Vehicle Number Plate Scanner")
st.write("Upload a cropped image of a number plate to extract the details.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Process Button
    if st.button('Extract Vehicle Number'):
        with st.spinner('Scanning plate...'):
            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            
            # Get Result
            plate_number = extract_number(img_array)
            
            if plate_number:
                st.success(f"✅ Extracted Number: **{plate_number}**")
                
                # Bonus: Add a simulated "Database" lookup
                st.info(f"🔍 Searching records for {plate_number}...")
                st.json({
                    "Vehicle Number": plate_number,
                    "Status": "Active",
                    "Owner Type": "Registered"
                })
            else:
                st.error("❌ Could not read the number. Please try a clearer image.")

st.markdown("---")
st.caption("OS Project Module - Developed by Spoorthi & Team")