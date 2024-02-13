# OCR Project

## Overview
This project aims to extract text from images using various Optical Character Recognition (OCR) methods. The project consists of a `Dataset` folder with the original images in their original formats, and a folder called `JPEG_Dataset` that consists of the images in jpg format. The images are labelled from `1.jpg` to `132.jpg`.

## OCR Methods Tested
We tested a number of different OCR methods, including:

1. **GOCR**: A traditional OCR approach was used to set a baseline.
2. **Tesseract**: An out-of-the-box deep learning OCR model.
3. **ChatGPT Vision**: Used to directly convert the image to the required ingredients list.
4. **Final Model**: Our final model combines Tesseract with a data transformation pipeline to apply transformations to preprocess the image, and uses the OCR text from Tesseract to be processed by ChatGPT to extract the ingredient list.

## Scripts
The following scripts are part of the project:

### Scripts
1. Rename Files - basic python nb to rename and convert to jpeg all the image files.
2. tom_tess.py: This tesseract script opens each image in the `JPEG_Dataset` folder, uses pytesseract to do OCR on the image, and writes the image name and extracted text to a CSV file named `ocr_results.csv`, and a  condensed version of the extracted text to the first 100 characters saved to a CSV file named `ocr_results_condensed.csv`.