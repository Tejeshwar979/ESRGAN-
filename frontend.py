# stream lit 
from utils.model import Generator  
from torchvision import transforms 
import streamlit as st 
from PIL import Image
import torch 
import torch.nn as nn 
from torchvision.transforms import ToPILImage
import pandas as pd 

import numpy as np  

transform = transforms.Compose([
    transforms.ToTensor() , 
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

to_pil = ToPILImage()

device = torch.device("cuda") 

generator = Generator().to(device)

checkpoint = torch.load(
    r"D:\ESRGAN\results\exp4\ESRGAN_save_4.pth" , 
    map_location = device
)

generator.load_state_dict(
    checkpoint["generator"]
)

generator.eval()


st.set_page_config(
    page_title = "Image Enhancer" , 
    layout = "wide"
)

st.title("ESRGAN Image Super Resolution") 

st.write("Upload a low resolution image to generate a High Resolution Image")

upload_file = st.file_uploader(
    "choose a low resolution image" , 
    type = ["png" , "jpg" , "jpeg"]
)

if upload_file is not None : 
    image = Image.open(upload_file)

    st.image(
        image , 
        caption = "low resolution image"
    )

    if st.button("generate high resolution") : 
        st.write("Generating...") 
    
        lr_tensor = transform(image) 
        lr_tensor = lr_tensor.unsqueeze(0)
        lr_tensor = lr_tensor.to(device)
        with torch.no_grad():
            hr_tensor = generator(lr_tensor)  
        
        hr_tensor = hr_tensor.squeeze(0).cpu() 
        hr_tensor = ( hr_tensor + 1 )/ 2

        hr_tensor = torch.clamp(hr_tensor ,0 , 1) 

        hr_image = to_pil(hr_tensor)

        st.image(
            hr_image , 
            caption = "high resolution image"
        )
        











