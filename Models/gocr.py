import subprocess
import os
import csv

# Set the directory containing your .jpeg images
image_folder_path = './JPEG_Dataset/'

# Specify the path for the output CSV file
output_csv_path = 'ocr_results_gocr.csv'

# Function to perform OCR using GOCR
def perform_ocr(image_path):
    # Command to run GOCR on the image
    command = ['gocr', image_path]
    # Execute the command
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Decode and return the output text
        return result.stdout.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        print(f"Error processing {image_path}: {e.stderr.decode('utf-8')}")
        return ""

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Image Name', 'Extracted Text'])

    # Loop through all the images in the folder
    for image_file in os.listdir(image_folder_path):
        if image_file.lower().endswith(".jpeg"):  # Adjust for other file extensions if necessary
            image_path = os.path.join(image_folder_path, image_file)
            # Perform OCR on the image using GOCR
            extracted_text = perform_ocr(image_path)
            # Write the image name and extracted text to the CSV
            writer.writerow([image_file, extracted_text])
            print(f"Processed {image_file}")

print(f"OCR results saved to {output_csv_path}")
