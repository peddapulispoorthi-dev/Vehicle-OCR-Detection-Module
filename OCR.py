import cv2
import easyocr
import re
import os

def extract_vehicle_number(image_path):
    # Check if file exists so the program doesn't crash
    if not os.path.exists(image_path):
        return "Error: Image file not found!"

    # 1. Load the Image
    img = cv2.imread(image_path)
    
    # 2. Preprocessing (Crucial for Indian Plates)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter removes noise but keeps edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    
    # Thresholding: Turns the image into strictly Black & White
    # This makes characters much easier for EasyOCR to read
    _, thresh = cv2.threshold(bfilter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. OCR
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have an NVIDIA card
    result = reader.readtext(thresh)
    
    if not result:
        return "No text detected"

    # 4. Post-processing
    raw_text = ""
    for detection in result:
        # detection[1] is the recognized text
        # detection[2] is the confidence score (useful for debugging!)
        raw_text += detection[1]
    
    # Clean the text: Uppercase and remove symbols/spaces
    clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    
    return clean_text

# --- Demo Testing ---
test_image = 'test_plate.jpg' 
print(f"Processing: {test_image}...")
result = extract_vehicle_number(test_image)
print(f"Final Output: {result}")