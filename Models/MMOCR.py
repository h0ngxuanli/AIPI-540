# Open anaconda prompt -> cd C:\Users\rs659\Desktop\AIPI 540
# mkdir venv -->
# conda create --prefix "C:\Users\rs659\Desktop\AIPI 540\venv" python=3.8
# conda activate "C:\Users\rs659\Desktop\AIPI 540\venv"
# conda install -p "C:\Users\rs659\Desktop\AIPI 540\venv" ipykernel --update-deps --force-reinstall
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# pip3 install openmim
# git clone https://github.com/open-mmlab/mmocr.git
# cd mmocr
# mim install -e .

from mmocr.apis import MMOCRInferencer
import os
import pandas as pd

class OCRProcessor:
    def __init__(self, det_model='DBNet', rec_model='SAR'):
        """Initialize the OCR model with specified detector and recognizer."""
        self.ocr = MMOCRInferencer(det=det_model, rec=rec_model)

    def process_images(self, image_dir):
        """Process all images in the given directory and return OCR results as a dictionary."""
        image_files = os.listdir(image_dir)
        ocr_text = {}
        for image in image_files:
            image_path = os.path.join(image_dir, image)
            print(f'Processing {image_path}')
            try:
                text = self.ocr(image_path, show=False)
                ocr_text[image] = ' '.join(text['predictions'][0]['rec_texts'])
            except Exception as e:
                print(e)
                ocr_text[image] = ''
        return ocr_text

    def save_results_to_csv(self, ocr_text, output_file='../Results/OCRData.csv'):
        """Save the OCR results to a CSV file."""
        df = pd.DataFrame.from_dict({'Images': list(ocr_text.keys()), 'Text-MMOcr': list(ocr_text.values())})
        df.to_csv(output_file, index=False)
        print(f'Results saved to {output_file}')

# Example usage
if __name__ == "__main__":
    ocr_processor = OCRProcessor(det_model='DBNet', rec_model='SAR')
    image_dir = '../Data/JPEG_Dataset/'
    ocr_results = ocr_processor.process_images(image_dir)
    ocr_processor.save_results_to_csv(ocr_results, '../Results/OCRData.csv')
