import torch 
import torch.nn as nn 


class fo(nn.Module) : 
    def __init__(self) : 
        super().__init__() 
        self.initial_features = nn.Conv2d(3 , 64 , 3 , 1, 1)
    def forward(self , x):
        return self.initial_features(x)

class denseBlock(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(64 , 32 , 3 , 1, 1)
        self.conv2 = nn.Conv2d(96 , 32 , 3 , 1 , 1)
        self.conv3 = nn.Conv2d(128 , 32 , 3 , 1 , 1)
        self.conv4 = nn.Conv2d(160 , 32 , 3 , 1, 1)
        self.conv5 = nn.Conv2d(192 , 64 , 3 , 1, 1)
        self.LReLU = nn.LeakyReLU(0.2) 
    def forward(self , x):
        x1 = self.LReLU(self.conv1(x)) 
        x2 = self.LReLU(self.conv2(torch.cat([x , x1] , dim = 1)))
        x3 = self.LReLU(self.conv3(torch.cat([x , x1 , x2] , dim = 1)))
        x4 = self.LReLU(self.conv4(torch.cat([x , x1 , x2 , x3] , dim = 1)))
        x5 = (self.conv5(torch.cat([x , x1 , x2 , x3 , x4] , dim = 1)))  
        return x + (0.2) * x5

class RRDB_Layer(nn.Module):
    def __init__(self):
        super().__init__() 
        self.dense_layer_1 = denseBlock()
        self.dense_layer_2 = denseBlock()
        self.dense_layer_3 = denseBlock()  
    def forward(self , x):
        out = self.dense_layer_1(x)
        out = self.dense_layer_2(out)
        out = self.dense_layer_3(out) 
        return x + (0.2) * out 

class RRDB_23_layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.RRDB_layer = nn.Sequential(
            *[RRDB_Layer() for _ in range(23)]
        )
    def forward(self , x):
        return self.RRDB_layer(x)

class lr_conv(nn.Module):
    def __init__(self):
        super().__init__() 
        self.convu = nn.Conv2d(64 , 64 , 3 , 1, 1)
    def forward(self , x):
        return self.convu(x)

class Generator(nn.Module) : 
    def __init__(self):
        super().__init__() 
        self.f0 = fo()
        self.RRDB_Layers_23 = RRDB_23_layers()
        self.lr_conv = lr_conv()
        self.upsampling1 = nn.Conv2d(64 , 256 , 3 , 1, 1)
        self.pixelshuffle = nn.PixelShuffle(2) 
        self.hr_convo1 = nn.Conv2d(64 , 64 , 3 , 1, 1)
        self.hr_convo2 = nn.Conv2d(64 , 3 , 3 , 1, 1)
        self.LReLU = nn.LeakyReLU(0.2)
    def forward(self , x):
        x0 = self.f0(x)
        out = self.RRDB_Layers_23(x0)
        out = self.lr_conv(out)
        part1_features = x0 + out
        # upsampling and hr-convo 
        out =  self.upsampling1(part1_features) 
        out =  self.pixelshuffle(out)
        out =  self.LReLU(out)
        out =  self.upsampling1(out) 
        out =  self.pixelshuffle(out)
        out =  self.LReLU(out)
        out =  self.hr_convo1(out)
        out =  self.LReLU(out)
        out =  self.hr_convo2(out)
        out =  self.LReLU(out)
        return out 

# Discriminator    

class Discriminator(nn.Module) : 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 , 64 , 3 , 1, 1) , 
            nn.LeakyReLU(0.2),
            nn.Conv2d(64 , 64 , 3 , 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64 , 128 , 3 , 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 , 128 , 3 , 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128 , 256 , 3 , 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 ,256 , 3 , 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256 , 512 , 3 , 1 , 1),
            nn.LeakyReLU(0.2) ,
            nn.Conv2d(512 , 512 , 3 , 2 ,1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100 , 1),
        )
    def forward(self , x):
        return self.model(x)


