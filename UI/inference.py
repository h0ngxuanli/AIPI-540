from openai import OpenAI
from PIL import Image
import pytesseract
# from dotenv import load_dotenv
from dotenv import dotenv_values


config = dotenv_values(".env")


def llm_postprocess(ocr_text):
    prompt = """
            Here is the extracted text from a product' ingredient part. 
            Could you please identify the ingredients and return the potential allergy ingredient? 
            Your return only includes the alleries with bullet points.
            """
    print(config["API_KEY"])
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


def tesseract_model(image_path):
    image = Image.open(image_path)
    ocr_text = pytesseract.image_to_string(image)
    allergies = llm_postprocess(ocr_text)
    return allergies, ocr_text



