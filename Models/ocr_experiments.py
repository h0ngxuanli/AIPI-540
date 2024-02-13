from openai import OpenAI
import openai
import base64
import json
import os
import pandas as pd
from PIL import Image
import wandb 
from tqdm import tqdm
from skimage.filters import threshold_local
import torch
import easyocr
import cv2
import pytesseract
from PIL import Image
import time
import subprocess
from mmocr.apis import MMOCRInferencer
from dotenv import dotenv_values


config = dotenv_values(".env")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def encode_image(image_path):
    
    """
    Encode an image as a base64 string.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: A base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def img_preprocess(image_path):
    
    """
    Preprocess an image for OCR by converting it to grayscale and applying a local threshold.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image suitable for OCR.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

def llm_postprocess(ocr_text):
    
    """
    Process OCR text using a large language model to identify potential allergy ingredients.

    Parameters:
        ocr_text (str): Text extracted from OCR.

    Returns:
        str: Identified potential allergy ingredients.
    """
    
    prompt = """
            Here is the extracted text from a product' ingredient part. 
            Could you please identify the ingredients and return the potential allergy ingredient? 
            Your return only includes the alleries with bullet points.
            """

    client = OpenAI(api_key=config["API_KEY"])
    
    response = client.chat.completions.create(
        model='gpt-4-vision-preview', #gpt-4
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": ocr_text},
                ],
            }
        ],
        max_tokens=800,
    )
    allergies = response.choices[0].message.content
    
    return allergies


def easyocr_model(image_path):
    reader = easyocr.Reader(['en'], gpu=True) # initialize OCR
    result = reader.readtext(image_path) # input image
    ocr_text =  "\n".join([res[1] for res in result])
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies

def easyocr_model_w_pre(image_path):
    reader = easyocr.Reader(['en'], gpu=True) # initialize OCR
    image = img_preprocess(image_path)
    result = reader.readtext(image) # input image
    ocr_text =  "\n".join([res[1] for res in result])
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies

def tesseract_model(image_path):
    image = Image.open(image_path)
    ocr_text = pytesseract.image_to_string(image)
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies

def tesseract_model_w_pre(image_path):
    image = Image.open(image_path)
    image = img_preprocess(image_path)
    # Use tesseract to do OCR on the image
    ocr_text = pytesseract.image_to_string(image)
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies

def mmocr_model(image_path):
    ocr = MMOCRInferencer(det='DBNet', rec='SAR', device='cuda:0')
    text = ocr(image_path, show=False)
    ocr_text= ' '.join(text['predictions'][0]['rec_texts'])
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies

def mmocr_model_w_pre(image_path):
    image_name = image_path.split("/")[-1]
    grey_image_path = "/home/featurize/work/OCR_exper/JPEG_Dataset_grey/" + image_name
    image = img_preprocess(image_path)
    cv2.imwrite(grey_image_path, image)
    
    ocr = MMOCRInferencer(det='DBNet', rec='SAR', device='cuda:0')
    text = ocr(grey_image_path, show=False)
    ocr_text= ' '.join(text['predictions'][0]['rec_texts'])
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies
    

def gocr_model(image_path):
    # Command to run GOCR on the image
    command = ['gocr', image_path]
    # Execute the command
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Decode and return the output text
        ocr_text =  result.stdout.decode('utf-8').strip()
    except:
        ocr_text =  " "
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies
    
def gocr_model_w_pre(image_path):
    # Command to run GOCR on the image
    image_name = image_path.split("/")[-1]
    grey_image_path = "/home/featurize/work/OCR_exper/JPEG_Dataset_grey/" + image_name
    image = img_preprocess(image_path)
    cv2.imwrite(grey_image_path, image)
    command = ['gocr', grey_image_path]
    # Execute the command
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Decode and return the output text
        ocr_text =  result.stdout.decode('utf-8').strip()
    except:
        ocr_text =  " "
    allergies = llm_postprocess(ocr_text)
    return ocr_text, allergies


def chatgpt_model(image_path):
    prompt = """
            Here is the picture from a product' ingredient part. 
            Could you please identify the ingredients and return the potential allergy ingredient? 
            Your return only includes the alleries with bullet points.
            """
    image_url = f"data:image/jpeg;base64,{encode_image(image_path)}"
    client = OpenAI(api_key=config["API_KEY"])

    response = client.chat.completions.create(
        model='gpt-4-vision-preview', #gpt-4
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ],
            }
        ],
        max_tokens=800,
    )
    allergies = response.choices[0].message.content
    ocr_text = " "
    return ocr_text, allergies
    


# initiate run
run = wandb.init(project = "aipi540", name = "OCR experiments")


# model collection
models = dict(zip(["chatgpt4", "easyocr", "gocr", "mmocr", "tesseract"],
                   [[chatgpt_model], [easyocr_model, easyocr_model_w_pre], 
                    [gocr_model, gocr_model_w_pre], [mmocr_model, mmocr_model_w_pre], 
                    [tesseract_model, tesseract_model_w_pre]]))
                   



data_dir = "/home/featurize/work/OCR_exper/JPEG_Dataset_synthetic/"

# create W&B table
table_cols = ["image", "path"] 
for model in models.keys():
    table_cols.append(model + " " + "text")
    table_cols.append(model + " " + "results")
    table_cols.append(model + " " + "time")
    if model != "chatgpt4":
        table_cols.append(model + "_preprocessed" + " " + "text")
        table_cols.append(model + "_preprocessed" + " " + "results")
        table_cols.append(model + "_preprocessed" + " " + "time")

table = wandb.Table(columns = table_cols)
artifact = wandb.Artifact(name="ocr_performance_comparsion", type="benchmark")


i = 1
for image_local in tqdm(os.listdir(data_dir)[:100]):
    
    try:
        image_array = wandb.Image(Image.open(data_dir + image_local))
    except:
        continue
    
    image_path  = data_dir + image_local
    
    table_data = [image_array, image_path]
    for model, model_func in models.items():
        try:
            start_time = time.time()
            ocr_text, ocr_result = model_func[0](image_path)
            end_time = time.time()
            ocr_time = end_time - start_time
        except:
            ocr_text = " "
            ocr_result = " " 
            ocr_time = " "
        table_data+=[ocr_text, ocr_result, ocr_time]
        
        if model != "chatgpt4":
            try:
                start_time = time.time()
                ocr_text, ocr_result = model_func[1](image_path)
                end_time = time.time()
                ocr_time = end_time - start_time
            except:
                ocr_text = " "
                ocr_result = " " 
                ocr_time = " "
            table_data+=[ocr_text, ocr_result, ocr_time]

    table.add_data(*table_data)
    
    
    if i%20 == 0:
        artifact.add(table, f"ocr benchmark #{i}")
        run.log_artifact(artifact)
        
        artifact = wandb.Artifact(name="ocr_performance_comparsion", type="benchmark")
        table_cols = ["image", "path"] 
        for model in models.keys():
            table_cols.append(model + " " + "text")
            table_cols.append(model + " " + "results")
            table_cols.append(model + " " + "time")
            if model != "chatgpt4":
                table_cols.append(model + "_preprocessed" + " " + "text")
                table_cols.append(model + "_preprocessed" + " " + "results")
                table_cols.append(model + "_preprocessed" + " " + "time")

        table = wandb.Table(columns = table_cols)
        
    i+=1

run.finish()