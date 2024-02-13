import pytesseract
import os
from PIL import Image

# Specify the directory containing your JPEG images
image_folder_path = './JPEG_Dataset/'

# Output file to save the OCR results
output_file_path = 'tess3_results.txt'

# If necessary, specify the path to your Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example for Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Example for Linux

# Function to perform OCR using Tesseract 3
def ocr_image(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(img, lang='eng', config='--oem 0')  # oem 0 for Tesseract 3
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process each image in the directory and write results to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for image_file in os.listdir(image_folder_path):
        if image_file.lower().endswith((".jpg", ".jpeg")):  # Adjust for other image formats if necessary
            image_path = os.path.join(image_folder_path, image_file)
            ocr_text = ocr_image(image_path)
            if ocr_text:
                output_file.write(f"OCR Results for {image_file}:\n{ocr_text}\n")
                output_file.write("-" * 80 + "\n")  # Separator between image results

print(f"OCR results have been saved to {output_file_path}")
