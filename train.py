from utils.loader import *  
from utils.model import *  

from torchvision import transforms 

import torch.optim as optim 

import argparse 

import gc
import torch
import torch.nn as nn  

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--data" , type = str  , default = r"D:\DIV2K_train_HR\DIV2K_train_HR")
parser.add_argument("--epochs" , type = int , default = 2) 
parser.add_argument("--save_model" , type = str, default = "ESRGAN_save_1.pth") 
parser.add_argument("--batch_size" , type = int , default = 4) 


args = parser.parse_args()

datasets = image_processing(root_dir_path=args.data)

train_dataloader = DataLoader(datasets , shuffle = True , batch_size = args.batch_size)

G = Generator().to(device)

D = Discriminator().to(device)

optimizer_g = optim.Adam(G.parameters()) 
optimizer_d = optim.Adam(D.parameters()) 

pixel_loss = nn.L1Loss()
adv_loss = nn.BCEWithLogitsLoss()



if __name__ == "__main__" : 
    G.train()
    D.train()
    epochs = args.epochs
    g_loss = 0.0 
    d_loss = 0.0 
    for epoch in range(epochs):
        for lr , hr in train_dataloader : 
                lr = lr.to(device)
                hr = hr.to(device) 

                real_labels = torch.ones(lr.size(0) , 1).to(device)

                fake_labels = torch.zeros(lr.size(0) , 1).to(device)

                # Generator train 

                output = G(lr)  

                pixel_ls = pixel_loss(output , hr) 

                pred_real_or_fake = D(output.detach())

                log_loss = adv_loss(pred_real_or_fake , real_labels)  

                mean_loss = pixel_ls + 1e-3 * log_loss  

                optimizer_g.zero_grad()

                mean_loss.backward()

                optimizer_g.step()

                # discriminator  train 

                real_pred = D(hr)  
                fake_pred = D(lr)

                real_loss = adv_loss(real_pred , real_labels)
                fake_loss = adv_loss(fake_pred , fake_labels)

                avg_loss = (real_loss + fake_loss) / 2  

                optimizer_d.zero_grad()

                avg_loss.backward() 

                optimizer_d.step()     

                g_loss += mean_loss
                d_loss += avg_loss 
        print(f"{epoch + 1} / {epochs}  g_loss = {g_loss/len(train_dataloader)} d_loss = {d_loss / len(train_dataloader)}")

    torch.save(
        {
            "generator" :    G.state_dict() , 
            "discrimintaor" : D.state_dict() , 
        },
        f"{args.save_model}"
    )


