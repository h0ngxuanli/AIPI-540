import os
from PIL import Image
import pytesseract
import csv

class OCRProcessor:
    def __init__(self, image_folder_path, output_csv_path):
        self.image_folder_path = image_folder_path
        self.output_csv_path = output_csv_path

    def process_images(self):
        """Process all images in the specified folder and save results to a CSV file."""
        with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Image Name', 'Extracted Text'])  # Write the header row
            for image_file in os.listdir(self.image_folder_path):
                if image_file.endswith(".jpeg"):  # Change this to match your file type
                    self.process_single_image(image_file, writer)
        print(f"OCR results saved to {self.output_csv_path}")

    def process_single_image(self, image_file, writer):
        """Process a single image, extract text using OCR, and write to the CSV."""
        image_path = os.path.join(self.image_folder_path, image_file)
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            writer.writerow([image_file, text])
            print(f"Processed {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Usage example
if __name__ == "__main__":
    image_folder_path = '../Data/JPEG_Dataset/'  # Set the path to your images
    output_csv_path = '../Results/tess_results.csv'  # Define the path for your output CSV file
    ocr_processor = OCRProcessor(image_folder_path, output_csv_path)
    ocr_processor.process_images()
