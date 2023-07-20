
from new import *
import streamlit as st
import torch

from PIL import Image
import os
import time
import numpy as np
from torch import nn
import pickle
import tqdm
import shutil
import webcolors

from east import *
from trocr import *
from data_preprocessing import *
from params import *
from dataset import *
from dataset import TextDataset, TextDatasetval
from util import *
from BigGAN_layers import *
from BigGAN_networks import *
from Discriminator import *
from generator import *
from transformer import *
from OCR_network import *
from blocks import *
from networks import *

from model import SLRGAN

model_path = r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\model.pth"  # Update with your model path
@st.cache_data
def generate_images( image, text_query):
    # Define the necessary paths and parameters
    output_path = 'results'
    batch_size = 1
   
    
    # Load the saved model
    model = SLRGAN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    print(model_path + ': Model loaded Successfully')

    # Convert the text query into encoded format
    text_encode = [j.encode() for j in text_query.split(' ')]
    eval_text_encode, eval_len_text = model.netconverter.encode(text_encode)
    eval_text_encode = eval_text_encode.to('cpu').repeat(batch_size, 1, 1)

    # Create the output directory if it doesn't exist
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

    # Process the uploaded image
    east = EAST(image)
    trocr = ImageProcessor()
    image_dict = trocr.process_images()
    cropped_img = r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\crop_images"

    # Create the dataset and data loader
    TestObj = Test(image_dict)
    datasetval = torch.utils.data.DataLoader(
        TestObj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=TestObj.collate_fn
    )

    generated_images = []
    for i, data_val in enumerate(tqdm.tqdm(datasetval)):
        # Generate the handwriting transformation for each image in the dataset
        page_val = model._generate_page(data_val['simg'].to('cpu'), data_val['swids'], eval_text_encode, eval_len_text)

        # Save the generated image
        filename = os.path.join(output_path, 'image' + str(i) + '.png')
        cv2.imwrite(filename, (page_val * 255).astype(np.uint8))

        # Append the generated image to the list
        generated_images.append(Image.open(filename))

    return generated_images
@st.cache_data
def generate_modified_image(image_path, B, G, R):
    image = cv2.imread(image_path)

    lower_range = np.array([0, 0, 0])  # Lower range of RGB values
    upper_range = np.array([220, 220, 220])  # Upper range of RGB values

    # Define the new RGB values
    new_rgb = np.array([B, G, R])  # New RGB values

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Get the current pixel's RGB values
            current_rgb = image[i, j]

            # Check if the current pixel's RGB values fall within the desired range
            if np.all(current_rgb >= lower_range) and np.all(current_rgb <= upper_range):
                image[i, j] = new_rgb

    image = cv2.bilateralFilter(image, 59, 100, 100)

    return image
background_images = {
    "White Texture": r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\white-texture.jpg",
    "Old paper ": r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\download.jpeg",
    "Pastel Mint Green Wrinkled Paper" : r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\green wrinkled background.jpg",
    "Dark yellow ": r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\download (1).jpeg"
}
@st.cache_data
def extract_pixels(source_image, target_image, target_range):
    # Load the source image
    source = cv2.imread(source_image)

    # Load the target image
    target = cv2.imread(target_image)

    # Resize the target image to match the dimensions of the source image
    target = cv2.resize(target, (source.shape[1], source.shape[0]))

    # Create a mask with white pixels where the RGB values are within the target range
    lower_range = np.array(target_range[0])
    upper_range = np.array(target_range[1])
    mask = cv2.inRange(source, lower_range, upper_range)

    # Extract the pixels by copying the source image using the mask
    extracted_pixels = cv2.bitwise_and(source, source, mask=mask)

    # Resize the extracted pixels to match the dimensions of the target image
    extracted_pixels = cv2.resize(extracted_pixels, (target.shape[1], target.shape[0]))

    # Replace the corresponding pixels in the target image with the extracted pixels
    result = np.where(mask[..., None], extracted_pixels, target)

    return result


def main():
    # Set page title and layout
    st.set_page_config(
        page_title="Handwriting Transformers",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    # Add the sidebar
    st.sidebar.title("IITI SOC Project")
    nav_selection = st.sidebar.radio("Go to", ("Home", "About Us"))

    # Page content
    if nav_selection == "Home":
        image_path = r"C:\Users\Lovely Bhardwaj\OneDrive\Desktop\models\file\content\ALL_FILES\Memorable design1.png"
        image = Image.open(image_path)
        st.image(image, use_column_width=True)

       
        st.markdown('<p style=" font-size: 16px; color: #FFFFFF;">Enter Text</p>', unsafe_allow_html=True)
        text_query = st.text_input('enter', key="text_input", help="Enter text here", label_visibility="collapsed")  

        
        st.markdown('<p style="font-size: 16px; color: #FFFFFF;">Upload an image here </p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],label_visibility="collapsed")

        if uploaded_file is not None:
            # Read the uploaded image data as bytes
            image_bytes = uploaded_file.getvalue()

            # Save the uploaded image to a temporary file
            temp_image_path = "temp_image.png"
            with open(temp_image_path, "wb") as f:
                f.write(image_bytes)

            # Process the uploaded image and generate handwriting
            generated_images = generate_images(temp_image_path, text_query)
            generated_image_paths = []
            for i, image in enumerate(generated_images):
                generated_image_path = f"generated_image_{i+1}.png"
                image.save(generated_image_path)  # Save generated image to a file
                generated_image_paths.append(generated_image_path)
                st.image(image, caption=f"Generated Image {i+1}")
                download_button_str = f"Download Image"
                st.download_button(download_button_str, data=image_bytes, file_name=f"generated_image_{i+1}.png")
                st.markdown('<p style="font-size: 16px; color: #FFFFFF;">Select Background paper</p>', unsafe_allow_html=True)
                target_image_name = st.selectbox("Background Image", list(background_images.keys()))

                # Perform image extraction only if a target image is selected
                if st.button("Get Background"):
                    target_image_path = background_images[target_image_name]
                    extracted_image = extract_pixels(generated_image_paths[0], target_image_path, [(0, 0, 0), (220, 220, 220)])
                    st.image(extracted_image, caption="New background")
                    download_button_str = f"Download Image"
                hex_color = st.color_picker("Select color", "#000000")
                rgb_color = webcolors.hex_to_rgb(hex_color)
                R, G, B = rgb_color
                if st.button("Get color"):
                    modified_image = generate_modified_image(generated_image_paths[0], B, G, R)
                    st.image(modified_image, channels="BGR", caption="Modified Image")
                    download_button_str = f"Download Image"
            os.remove(temp_image_path)
            
        
            

    elif nav_selection == "About Us":
        st.title("About Us")
        # st.markdown('<p style="text-align: center; font-family: Times New Roman, serif; font-size: 36px; color: #00FF00;">About Handwriting Transformers Team</p>', unsafe_allow_html=True)
        st.write("Welcome to the About Us page!")
        st.write("Here you can find information about the Handwriting Transformers team and project.")

        # Add additional details for the About Us page...

if __name__ == '__main__':
    main()



