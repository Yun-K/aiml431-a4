"""It contains some user defined functions such as loading, preprocessing, and Data augmentation.

"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# TODO:!!! Directory containing the data.
#
# root = 'YOUR_PATH_TO_TRAIN_DATA'

# the root directory that contains the data
root = "emnist_jpeg/EMNIST"


def get_EMNIST(params):
    #TODO:! Part 1. Complete the data_loader and ensure it can successfully load images from your folder
    transform = transforms.Compose([ 
        transforms.Grayscale(),
        # TODO:! Add some processing transforms for data augmentation
        transforms.Resize(params['imsize']),
        transforms.ToTensor(),
        
        ])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader