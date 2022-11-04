"""It contains some user defined functions such as loading, preprocessing, and Data augmentation.

"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision.utils as vutils
import numpy as np


random.seed(0)
torch.manual_seed(0)

# the root directory that contains the data
# 6, 1, 2 subfolders are copied to the Train folder
root = "emnist_jpeg/Train"


# # random choose a subfolder
# def get_random_subfolder(root):
#     subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
#     return random.choice(subfolders)
# # root = get_random_subfolder(root)
# # print(root)

def get_EMNIST(params):
    # Part 1. Complete the data_loader and ensure it can successfully load images from your folder 
    # https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99
    transform = transforms.Compose([ 
        transforms.Grayscale(),
        # Add some processing transforms for data augmentation
        transforms.Resize(params['imsize']), # resize the image to 32x32
               
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(), # randomly flip the image vertically
        transforms.RandomRotation((90,90)) ,
        transforms.RandomVerticalFlip(1),       

        # # based on the image center, randomly rotate the image by 20 degrees, translate the image by 0-0.1 in x and y directions, and scale the image by 0.8 to 1.2 times
        # transforms.RandomAffine(20, translate=(0,0.1), scale=(0.8,1.2)), 
        
        
        
        # transforms.ColorJitter(brightness=0.2, contrast=0.2), # randomly change the brightness, contrast 
        # normalize image with mean and standard deviation,
        # which means (x-mean)/std
        transforms.ToTensor(), # transform it into a torch tensor
        transforms.Normalize((0.5,), (0.5,)), # normalize to [-1, 1]
        ])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

        
    # print some information about the dataset 
    print('-'*60)
    print(f"Dataset: {dataset}")
    print('-'*60)
    
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # plot some training images
    def plot_images():
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to("cpu")[:32], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
    # plot_images()
    
    return dataloader


if __name__ == '__main__':
    # from train import params
    print(root)
    # from train import load_model_and_generate_images
    # root_loader = get_EMNIST({'imsize' : 32, 'bsize' : 32})