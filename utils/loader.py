from PIL import Image  
from torch.utils.data import DataLoader , datasets
import os  
from torchvision import transforms  

class image_processing():
    def __init__(self , root_dir_path ):
        self.root_dir_path = root_dir_path 
        self.images_data = [os.path.join(img_path , root_dir_path) for img_path in os.listdir(self.root_dir_path) ]
        self.transformations = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5)),
        ]) 
    def __len__(self): 
        return len(self.images_data)
    def __getitem__(self , idx):
        img_path = self.images_data[idx] 
        img = Image.open(img_path).convert("RGB")
        hr = img.resize((256 , 256) , Image.BICUBIC)
        hr = self.transformations(img)
        lr = self.resize((64 , 64) , Image.BICUBIC)
        lr = self.transformations(lr)
        return lr , hr  




