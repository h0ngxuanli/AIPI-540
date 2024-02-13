import streamlit as st
import os
import PIL as pil
import torch
from inference import llm_postprocess, tesseract_model
from PIL import ImageOps
from PIL import Image, ImageOps


device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if device == 'mps':
    torch.mps.empty_cache()


# best_model_btn = st.button("Load Best Models")

# if "best_model_btn_btn_state" not in st.session_state:
#     st.session_state.best_model_btn_btn_state = None

# if best_model_btn or st.session_state.best_model_btn_btn_state:
#     st.session_state.best_model_btn_btn_state = True


#     st.write("Models Loaded")


uploadbtn = st.button("Upload Image")

if "uploadbtn_state" not in st.session_state:
    st.session_state.uploadbtn_state = False

if uploadbtn or st.session_state.uploadbtn_state:
    st.session_state.uploadbtn_state = True

    image_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])
    
    # if image_file:
    #     image_file = ImageOps.exif_transpose(image_file)

    if image_file:
        img = Image.open(image_file)
        img = ImageOps.exif_transpose(img)
        # st.image(img)
        st.image(img, caption='Image for Prediction')

    # if image_file is not None:
    #     st.image(image_file, caption='Image for Prediction')



extractTextbtn = st.button("Get Allergies")

if "extractTextbtn_state" not in st.session_state:
    st.session_state.extractTextbtn_state = False

if extractTextbtn or st.session_state.extractTextbtn_state:
    st.session_state.extractTextbtn_state = True

    allergies = tesseract_model(image_file)

    st.text(allergies)

    # llm_postprocess(ocr_text)(image_file)

    # image_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])

    # if image_file is not None:
    #     st.image(image_file, caption='Image for Prediction')


    #     org_image = pil.Image.open(image_file, mode='r')
    #     st.text("Uploaded image")
    #     st.image(org_image, caption='Image for Prediction')
    #     pred_button = st.button("Perform Prediction")
    #     if pred_button:
    #         st.image(org_image, caption='Predicted Image')

    #         # for model in model_loaded_list:
    #         for model_name, model in model_loaded_list.items():

    #             predicted_class, predicted_class_name = predict(model,org_image,device)
    #             st.write(f"The class predict by {model_name} is  : {predicted_class_name}")
    #             result = get_torch_cam(model,model_name,org_image)
    #             st.image(result, caption=f'{model_name }Predicted Heat Map Image')