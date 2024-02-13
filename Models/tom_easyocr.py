import easyocr
import os
import csv

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Add additional languages as needed

# Set the directory containing your .jpeg images
image_folder_path = './JPEG_Dataset/'

# Specify the path for the output CSV file
output_csv_path = 'ocr_results_easyocr.csv'

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Image Name', 'Extracted Text'])

    # Loop through all the images in the folder
    for image_file in os.listdir(image_folder_path):
        if image_file.endswith(".jpeg"):  # Adjust if you're using another file extension
            image_path = os.path.join(image_folder_path, image_file)
            try:
                # Perform OCR on the image
                results = reader.readtext(image_path)
                # Concatenate all detected text pieces into one string
                extracted_text = " ".join([text for _, text, _ in results])
                # Write the image name and extracted text to the CSV
                writer.writerow([image_file, extracted_text])
                print(f"Processed {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

print(f"OCR results saved to {output_csv_path}")
