import pytesseract
import os
from PIL import Image

class TesseractOCR:
    def __init__(self, image_folder_path, output_file_path):
        self.image_folder_path = image_folder_path
        self.output_file_path = output_file_path
        # Uncomment and set the path according to your Tesseract installation
        # pytesseract.pytesseract.tesseract_cmd = r'Path_To_Tesseract_Executable'

    def ocr_image(self, image_path):
        """Perform OCR on a single image and return the text."""
        try:
            img = Image.open(image_path)
            # Using Tesseract 3 configuration
            text = pytesseract.image_to_string(img, lang='eng', config='--oem 0')
            return text
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_images(self):
        """Process all images in the directory and write OCR results to a file."""
        with open(self.output_file_path, 'w', encoding='utf-8') as output_file:
            for image_file in os.listdir(self.image_folder_path):
                if image_file.lower().endswith((".jpg", ".jpeg")):  # Adjust for other formats if necessary
                    image_path = os.path.join(self.image_folder_path, image_file)
                    ocr_text = self.ocr_image(image_path)
                    if ocr_text:
                        output_file.write(f"OCR Results for {image_file}:\n{ocr_text}\n")
                        output_file.write("-" * 80 + "\n")  # Separator between results
        print(f"OCR results have been saved to {self.output_file_path}")

# Example usage
if __name__ == "__main__":
    image_folder_path = '../Data/JPEG_Dataset/'
    output_file_path = '../Results/tess3_results.txt'
    ocr = TesseractOCR(image_folder_path, output_file_path)
    ocr.process_images()
