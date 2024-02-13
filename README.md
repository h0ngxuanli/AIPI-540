# OCR Project

## Project Structure
The project is organized into four main directories: `Models`, `Results`, `UI`, and `Notebooks`.

- `Models`: Contains the models for each OCR method tested.
- `Results`: Stores the output from each of the OCR methods.
- `UI`: Houses the Streamlit `UI.py` file that runs a web server.
- `Notebooks`: Contains notebooks for each OCR method.

## Overview
This project focuses on the extraction of text from images using various Optical Character Recognition (OCR) methods. The goal is to accurately and efficiently convert image-based text into machine-readable text. The OCR method extracts the text, and our web server component uses ChaTGPT to extract the relevant information (allergies and dietary restrictions) from the text.

## OCR Methods
We have explored and implemented several OCR methods in this project:

1. **GOCR**: A traditional OCR approach that provides a baseline for our experiments.
2. **Tesseract**: A popular OCR engine that uses deep learning techniques.
3. **ChatGPT Vision**: A novel approach that directly converts images to text.
4. **EasyOCR Model**: A model from the EasyOCR library that we have fine-tuned on our dataset.
5. **Final Model**: Our custom model that combines Tesseract and a data transformation pipeline. This model preprocesses the image, applies OCR using Tesseract, and then processes the OCR text with ChatGPT to extract the ingredient list.

Each of these methods has corresponding files in the `Models` and `Notebooks` folders.

## User Interface
We have developed a user interface using Streamlit. The `UI.py` file in the `UI` folder runs a web server that deploys our basic pipeline. Users can upload images and receive back an allergy list.

## Results
The `Results` folder contains the output from each of the OCR methods tested. This allows us to compare the performance of each method and make informed decisions about which methods to use or further develop.

## Weights & Biases Integration
We have documented our testing, tabulation, reports, and comparisons between our different approaches on the Weights & Biases platform. You can view our project here: https://wandb.ai/aipi549/AIPI-540-classifier 