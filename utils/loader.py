from PIL import Image  
from torch.utils.data import DataLoader 
import os  
from torchvision import transforms  

class image_processing():
    def __init__(self , root_dir_path ):
        self.root_dir_path = root_dir_path 
        self.images_data = [os.path.join(root_dir_path , img) for img in os.listdir(self.root_dir_path) ]
        self.transformations = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5)),
        ]) 
    def __len__(self): 
        return len(self.images_data)
    def __getitem__(self , idx):
        img_path = self.images_data[idx] 
        hr = Image.open(img_path).convert("RGB")
        hr = hr.resize((256 , 256) , Image.BICUBIC)
        hight_res = self.transformations(hr)
        lr = hr.resize((64 , 64) , Image.BICUBIC)
        low_res = self.transformations(lr)
        return low_res , hight_res  




