from matplotlib import animation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from utils import get_EMNIST
from dcgan import weights_init, Generator, Discriminator
from IPython.display import HTML, display
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display
from matplotlib.animation import FuncAnimation, PillowWriter
import torchvision.utils as vutils

# Random seeds to ensure reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define for the program
#TODO:!!! Change the parameters according to your own needs
params = {
    "bsize" : 32,# Batch size during training.
    'imsize' : 32,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 1,# Number of channles in the training images. For greyscale images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    # 'nepochs' : 2,# Number of training epochs.
    'nepochs' : 150,# Number of training epochs.
    'lr' : 0.0001,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 10}# Save step.

# Use GPU if cuda is available, else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#device = torch.device("cpu")
#Use the above code if GPU is not available

# Get the data.
dataloader = get_EMNIST(params)
num_epochs = params['nepochs']

#The help function to plot the loss trend
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='DCGAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

#The help function for plotting results
def plot_result(generator, noise, num_epoch, save=False, save_dir='DCGAN_results/', show=True, fig_size=(5, 5), isTrainedModel=False):
    #TODO:!!! Call the model G (generator) properly to generate images from noises (Part 3.3)
    with torch.no_grad():# disable gradient calculation to avoid problems with the autograd engine
        gen_image = generator(noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_image, padding=2, normalize=True))
    
    
    generator.train()
    #Important! If you are calling this plot result function during training,
    # don't forget to change the model mode from eval() to train().

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    
    # print(f"n_rows: {n_rows}, n_cols: {n_cols}, noise.size(): {noise.size()} \n generated images number = {gen_image.size()[0]}\n")
    
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.adjustable = 'box-forced'
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img.astype(np.uint8), cmap='Greys_r') # The default grey scale visualization is reversed. Using "_r" to show the correct scale
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1) if not isTrainedModel else 'Images generated by the trained Model'
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

netG = Generator(params).to(device)
# The weights_init() function is called to randomly initialize all weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model to check the structure
print(netG)

netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)

# For defining the loss function to use: Binary Cross Entropy loss function.
criterion = nn.BCELoss()

#Define a fixed noise map to ensure the reproducibility of your results
fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# To Store G's and D's losses during training.
G_losses = []
D_losses = []

iters = 0



print("Now Training...")
print("-"*25)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # TODO:!!! Part 3: Complete code from here to finish the training/generation process of DCGAN

        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_data.size(0)

        # TODO:!!! training (a): train D using real data
        # Here, D_x = output.mean().item() can be used to denote the corresponding term of this part
        # errD_real is the loss generated here
        
        netD.zero_grad() # Zero the gradient buffers
        label = torch.full((b_size,), real_label, device=device, dtype=torch.float) # to avoid RuntimeError: Found dtype Long but expected Float
        # forward pass real batch data through D
        output = netD(real_data).view(-1)
        # calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        
        # TODO:!!! training (b): train D using fake data
        #  -- fake data needs to go through the current G.
        # Here, D_G_z1 = output.mean().item() can be used to denote the corresponding term of this part
        # Sample random data from a unit normal distribution.
        # errD_fake is the loss generated here

        # Generate batch of latent vectors
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # calculate D's loss on the all-fake batch
        # print(f"output shape: {output.shape}")
        # print(f"label_shape: {label.shape}")
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        #  accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # # get the err of D as the sum of the err of D_real and D_fake
        # errD = errD_real + errD_fake
        # optimizerD.step() # update Discriminator's parameters
        
        # Net discriminator loss.
        errD = errD_real + errD_fake
        # Update discriminator parameters.
        optimizerD.step()

        # TODO:!!! training (c): train G by backpropagation
        # Here, D_G_z2 = output.mean().item() can be used to represent the corresponding loss term.
        # errG is the overall loss for G
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # due to D has just been updated, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        errG = criterion(output, label) # get the loss of G based on the output
        errG.backward() # calculate the gradients of G
        D_G_z2 = output.mean().item() # get the mean of the output of D
        
        # 
        
        
        
        # Update generator parameters.
        optimizerG.step()

        # Check progress of training.
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    # Save the model.
    if epoch % params['save_epoch'] == 0 or epoch == params['nepochs']-1:
        #TODO:!!! Save the model (Part 3.2)
        torch.save(netG.state_dict(), "DCGAN_results/saved_models/netG_epoch_%d.pth" % (epoch))
        plot_loss(D_losses, G_losses, epoch, save=True)

    # Show result for fixed noise
    if epoch % (10 / 2) == 0 or (epoch == num_epochs-1):
        #TODO:!!! Finish the code in "plot_result()" and call it properly to generate and save images
        plot_result(netG, fixed_noise, epoch, save=True, show=False,fig_size=(5, 5))

# finish training, chaneg the mode back to eval mode to avoid the influence of dropout and batch normalization states
netG.eval()


# TODO:!!! Save the final trained model.
torch.save(netG.state_dict(), 'DCGAN_results/saved_models/netG_final.pth')

# TODO:Plot the training losses.
# Plot the training losses for the generator and discriminator, 
# taken from the G_losses and D_losses lists.
# these show the loss of G and D with respect to the number of training iterations.
plot_loss(D_losses, G_losses, num_epochs)

# Below is a plot of D & G’s losses versus training iterations.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('DCGAN_results/G_D_losses_verus_training_iterations.png')
plt.show()


# TODO:Animation showing the improvements of the generator.
def plot_animation():
    # Display the images generated by the generator as training progresses.
    # the animation will be saved in the DCGAN_results/DCGAN_training_animation.gif
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    # then, animate the images generated by the generator
    ims = [[plt.imshow(np.transpose(i, (1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    # save the animation, slower speed than the original video
    writergif = animation.PillowWriter(fps=3)
    ani.save('DCGAN_results/DCGAN_training_animation.gif', writer=writergif)
    
    # display the animation
    HTML(ani.to_jshtml())

# then, Real images and fake images side-by-side
def plot_side_by_side(fig_size=(15, 15)):
    """Generate fake images and plot them side-by-side with real images."""
    real_batch = next(iter(dataloader))
    plt.figure(figsize=fig_size)
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=5, normalize=True).cpu(),(1,2,0)))
    
    # plot fake images generated by the generator  from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images from the last epoch")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # save figure  FIXME: not saved successfully but it is displayed successfully
    plt.savefig('DCGAN_results/DCGAN_real_and_fake_images.png')
    
    plt.show()

# optional method for loading a trained model of the generator, and using it to generate images
def load_model_and_generate_images(netGPath, save_dir="DCGAN_results/generated_imgs_from_saved_model/"):
    """ This function loads a trained model of the generator, and uses it to generate images.

    Through my testing, it works as expected
    """
    # load the trained model
    trained_generator_model = Generator(params).to(device)
    trained_generator_model.load_state_dict(torch.load(netGPath))
    trained_generator_model.eval()
    # generate images by using trained_model 
    # and save them to save_dir
    plot_result(trained_generator_model, fixed_noise, 0, save=True, show=True,fig_size=(5, 5), save_dir=save_dir, isTrainedModel=True)

load_model_and_generate_images("DCGAN_results/saved_models/netG_final.pth")
# assert False, "Need to stop  as we need to generate images from the saved model"

plot_animation()
plot_side_by_side()



#TODO:!!! Part 4 (Challenge) Make necessary changes to adapt to Conditional DCGAN model.
#Any reasonable attempts you made could get some points, even the new model doesn't work properly.

# code is inspired from : https://towardsdatascience.com/using-conditional-deep-convolutional-gans-to-generate-custom-faces-from-text-descriptions-e18cc7b8821

class ConditionalGenerator(nn.Module):
    """Generator for Conditional DCGAN model.
    We have 2 classes of images, so we need to add a label to the input of the generator.
    """
    def __init__(self, params = params, classes=2):
        super(ConditionalGenerator, self).__init__()
        self.params = params
        self.classes = classes
        self.main = nn.Sequential(
            nn.ConvTranspose2d(params['nz']+self.classes, params['ngf'] * 4, 4, 2, 0, bias=False),
            nn.BatchNorm2d(params['ngf'] * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8 --> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(params['ngf'] * 4, params['ngf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf'] * 2),
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
        
    def forward(self, input, label):
        # concatenate the input and the label
        input = torch.cat((input, label), 1)
        output = self.main(input)
        return output
        
        
        
class ConditionalDiscriminator(nn.Module):
    """Discriminator for Conditional DCGAN model."""
    
    def __init__(self, params = params, classes=2):
        super(ConditionalDiscriminator, self).__init__()
        self.params = params
        self.classes = classes
        self.conv1 = nn.Sequential(
            # 1*32*32 --> 64*16*16
            nn.Conv2d(params["nc"]+self.classes, params["ndf"], 4, 2, 1, bias=False),
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
        
    def forward(self, input, label):
        # concatenate the input and the label
        input = torch.cat((input, label), 1)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output.view(-1, 1).squeeze(1)
    
    
    
