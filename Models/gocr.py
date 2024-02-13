import subprocess
import os
import csv

class GOCR:
    def __init__(self, image_folder_path='../Data/JPEG_Dataset/', output_csv_path='../Results/ocr_results_gocr.csv'):
        self.image_folder_path = image_folder_path
        self.output_csv_path = output_csv_path

    def perform_ocr(self, image_path):
        command = ['gocr', image_path]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return result.stdout.decode('utf-8').strip()
        except subprocess.CalledProcessError as e:
            print(f"Error processing {image_path}: {e.stderr.decode('utf-8')}")
            return ""

    def process_images(self):
        with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Image Name', 'Extracted Text'])

            for image_file in os.listdir(self.image_folder_path):
                if image_file.lower().endswith(".jpeg"):
                    image_path = os.path.join(self.image_folder_path, image_file)
                    extracted_text = self.perform_ocr(image_path)
                    writer.writerow([image_file, extracted_text])
                    print(f"Processed {image_file}")

        print(f"OCR results saved to {self.output_csv_path}")

if __name__ == "__main__":
    image_folder_path = '../Data/JPEG_Dataset/'  # Set the path to your images
    output_csv_path = '../Results/ocr_results_gocr.csv'  # Define the path for your output CSV file
    ocr_processor = GOCR(image_folder_path, output_csv_path)
    ocr_processor.process_images()
