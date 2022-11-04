import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


# Part 2. Finish the following two definition of classes to define Generator and Discriminitor model structure
# Define the Generator Network
class Generator(nn.Module):
    """The Generator network takes a latent vector z as input and outputs a generated image.
    The generator G consists of several deconvolutional layers to progressively upsample the input noise Z to generate an image which looks like an image from the real dataset.
    
    Structure:
    Input -> Deconvolutional -> BatchNorm -> ReLU -> Deconvolutional -> BatchNorm -> ReLU -> Deconvolutional -> Tanh -> Output
    
    Input: a 100-dimensional random vector z , H*W*C = 1*1*100
    Output: a single channel image of size 32*32 (H*W*C = 32*32*1)

    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#implementation
    """
    
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.

        #TODO:!! Define layers here
        self.main = nn.Sequential(
            # input is Z, going into a deconvolution
            # should be 100*1*1 --> 256*8*8
            nn.ConvTranspose2d(params["nz"], params["ngf"] * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(params["ngf"] * 4), # 256 = 4 * 64
            nn.ReLU(True),  
            # state size. (ngf*4) x 8 x 8 --> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(params["ngf"] * 4, params["ngf"] * 2, 4, 2, 1,  bias=False),
            nn.BatchNorm2d(params["ngf"] * 2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 16 x 16 --> (ngf) x 32 x 32
            nn.ConvTranspose2d(params["ngf"] * 2, params["ngf"], 4, 2, 1,  bias=False),
            nn.BatchNorm2d(params["ngf"]),
            nn.ReLU(True),
            
            # finally, the generated image should be H*W*C = 32*32*1
            # state size. (ngf) x 32 x 32 --> nc x 32 x 32  (nc = 1)
            nn.ConvTranspose2d(params["ngf"], params["nc"], 4, 2, 1,  bias=False),
            nn.Tanh(),
            
        )
        

    def forward(self, x):
        #TODO:!! Define how the data flows in Generator
        # print("-"*30)
        # print(f"Generator input shape: {x.shape}")
        output = self.main(x)
        # print some information
        # print(f"Generator output shape: {output.shape}")
        # print("-"*30)
        
        return output
class Discriminator(nn.Module):
    """input image size is 32x32, 
    the convolutional layers are defined as follows:
    32*32 input image, output channel Number: 64, kernel size is K=4, stride is S=2
    2. conv input size: 32*32 
    3. Batch Normalization
    4. ReLU
    5. Conv Input size: 16*16
    6. Batch Normalization
    7. ReLU
    8. Conv Input Size:8*8
    9. Batch Normalization
    10. ReLU
    11. Conv Input Size:4*4
    12. Sigmoid
    Then, Real or Fake is determined by the output of the discriminator where Real is 1 and Fake is 0.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#implementation
    """
    def __init__(self, params):
        super().__init__()
        # Input Dimension: (nc) x 32 x 32
        #TODO:!! Define layers here
        self.conv1 = nn.Sequential(
            # 1*32*32 --> 64*16*16
            nn.Conv2d(1, params["ndf"], 4, 2, 1, bias=False),
            nn.BatchNorm2d(params["ndf"]),
            nn.ReLU(True),
        ) 
        self.conv2 = nn.Sequential(
            # 64*16*16 --> 128*8*8
            nn.Conv2d(params["ndf"], params["ndf"]*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params["ndf"]*2),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            # 128*8*8 --> 256*4*4
            nn.Conv2d(params["ndf"]*2,params["ndf"]*2*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params["ndf"]*2*2), # 256
            nn.ReLU(True),
        )        
        self.conv4 = nn.Sequential(
            nn.Conv2d(params["ndf"]*2*2, 1, 4, 2, 0, bias=False),
            nn.Sigmoid(),
        )
        # we use the sigmoid function to get the probability of the image being real or fake
        # the output is a scalar between 0 and 1 where 0 is fake and 1 is real
        
        


    def forward(self, x):
        # print("-"*40)
        # print(f"Discriminator input shape: {x.shape}")

        #TODO:!! Define how the data flows in Discriminator
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(f"Discriminator output shape: {x.shape}")
        # print("-"*40)
        return x
