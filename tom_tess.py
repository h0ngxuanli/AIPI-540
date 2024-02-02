import os
from PIL import Image
import pytesseract
import csv

# Set the path to your images
image_folder_path = './JPEG_Dataset/'

# # Define the path for your output CSV file
# output_csv_path = 'ocr_results.csv'

# # Open or create the CSV file for writing
# with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # Write the header row
#     writer.writerow(['Image Name', 'Extracted Text'])

#     # Loop through all the images in the folder
#     for image_file in os.listdir(image_folder_path):
#         if image_file.endswith(".jpg"):
#             image_path = os.path.join(image_folder_path, image_file)
#             try:
#                 # Open the image
#                 img = Image.open(image_path)
#                 # Use pytesseract to do OCR on the image
#                 text = pytesseract.image_to_string(img)
#                 # Write the image name and extracted text to the CSV
#                 writer.writerow([image_file, text])
#                 print(f"Processed {image_file}")
#             except Exception as e:
#                 print(f"Error processing {image_file}: {e}")

# print(f"OCR results saved to {output_csv_path}")


# Define the path for your output CSV file
output_csv_path = 'ocr_results_condensed.csv'

# Open or create the CSV file for writing
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Image Name', 'Extracted Text (condensed)'])

    # Loop through all the images in the folder
    for image_file in os.listdir(image_folder_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_folder_path, image_file)
            try:
                # Open the image
                img = Image.open(image_path)
                # Use pytesseract to do OCR on the image
                text = pytesseract.image_to_string(img)
                # Condense the text to the first 100 characters
                condensed_text = text[:100] + (text[100:] and '...')
                # Write the image name and condensed extracted text to the CSV
                writer.writerow([image_file, condensed_text])
                print(f"Processed {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

print(f"OCR results saved to {output_csv_path}")
