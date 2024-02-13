import easyocr
import os
import csv

class EasyOCR:
    def __init__(self, image_folder_path='../Data/JPEG_Dataset/', output_csv_path='../Results/ocr_results_easyocr.csv'):
        self.reader = easyocr.Reader(['en'])  # Add additional languages as needed
        self.image_folder_path = image_folder_path
        self.output_csv_path = output_csv_path

    def perform_ocr(self, image_path):
        try:
            results = self.reader.readtext(image_path)
            extracted_text = " ".join([text for _, text, _ in results])
            return extracted_text
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""

    def process_images(self):
        with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Image Name', 'Extracted Text'])

            for image_file in os.listdir(self.image_folder_path):
                if image_file.endswith(".jpeg"):
                    image_path = os.path.join(self.image_folder_path, image_file)
                    extracted_text = self.perform_ocr(image_path)
                    writer.writerow([image_file, extracted_text])
                    print(f"Processed {image_file}")

        print(f"OCR results saved to {self.output_csv_path}")

# Usage
#easyocr = EasyOCR()
#easyocr.process_images()
