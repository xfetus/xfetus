from os import path as osp
import random

from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import wandb
from diffusers import DDIMScheduler, DDPMPipeline
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str

from sr_dataset import FetalPlaneDataset
from hat_utils import parse_options

if __name__ == "__main__":

    ##################
    ##   1. SETUP   ##
    ##################

    # Command line aurgments - for script
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="File location of the fetal brain dataset", type=str)
    parser.add_argument("-w", "--wandb_enabled", help="Enable weights and bias logging", type=bool)
    #parser.add_argument("-p", "--ddpm_checkpoint_path", help="File location for the pre-trained DDPM model", type=str)
    parser.add_argument("-i", "--hat_checkpoint_path", help="File location for the pre-trained HAT model", type=str)
    parser.add_argument("-c", "--hat_config_path", help="File location for the config yaml for HAT model", type=str)
    args = parser.parse_args()
    wandb_enabled = args.wandb_enabled
    dataset_path = args.dataset_path
    #ddpm_checkpoint_path = args.ddpm_checkpoint_path
    hat_checkpoint_path = args.hat_checkpoint_path
    hat_config_path = args.hat_config_path
    # Command line aurgments - for google colab
    '''wandb_enabled = False
    dataset_path = '/content/FETAL_PLANES_ZENODO/'
    ddpm_checkpoint_path = "/content/gdrive/MyDrive/128xfetal_5000.pth"
    hat_config_path = '/content/gdrive/MyDrive/HAT_SRx4_ImageNet-pretrain.yml'
    hat_checkpoint_path = "/content/gdrive/MyDrive/HAT_SRx4_ImageNet-pretrain.pth"'''

    # start a new wandb run to track this script
    if wandb_enabled:
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",
            # track hyperparameters and run metadata
            config={
                "architecture": "CNN",
                "dataset": "Fetal Plane dataset",
            }
        )

    # Define path to the dataset ('Scratch/FETAL_PLANES_ZENODO/' on Myriad)
    image_path = dataset_path + "Images"
    csv_path = dataset_path + "FETAL_PLANES_DB_data.csv"

    # Are we using a GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Filter dataset 
    plane = None # 'Fetal brain', 'Fetal thorax', 'Maternal cervix', 'Fetal femur', 'Fetal thorax', 'Other'
    operator_number = None # 'Op. 1', 'Op. 2', 'Op. 3', 'Other'
    us_machine = 'Voluson E6' # None, 'Voluson S10'
    brain_plane = None # 'Trans-cerebellum', 'Trans-thalamic', 'Trans-ventricular'   

    # Define hyperparameters
    image_size = 512
    batch_size = 1
    epochs = 300
    learning_rate = 1e-5  

    ####################
    ##   2. DATASET   ##
    ####################


    # Description of how each image in the dataset will be transformed
    transform_operations=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(45)
                      ])


    # Define the dataset for fine tuning the DDPM
    training_dataset = FetalPlaneDataset(root_dir=image_path,
                            csv_file=csv_path,
                            plane=None,
                            brain_plane=None,
                            us_machine='Voluson E6',
                            operator_number=None,
                            transform=transform_operations,  
                            train=1,
                            validation=False,
                            size=image_size,
                        )
    # Define corresponding dataloader
    training_dataloader = DataLoader(training_dataset, 
                        batch_size=batch_size,
                        shuffle=True)

    validation_dataset = FetalPlaneDataset(root_dir=image_path,
                            csv_file=csv_path,
                            plane=None,
                            brain_plane=None,
                            us_machine='Voluson E6',
                            operator_number=None,
                            transform=transform_operations,  
                            train=0,
                            validation=True,
                            size=image_size,
                        )
    # Define corresponding dataloader
    validation_dataloader = DataLoader(validation_dataset, 
                        batch_size=batch_size,
                        shuffle=True)
    
    ##############################
    ##   3. MODEL & OPTIMIZER   ##
    ##############################

    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(hat_config_path, is_train=False)
    opt['path']['pretrain_network_g'] = hat_checkpoint_path
    torch.backends.cudnn.benchmark = True
    # create HAT superresolution model
    model = build_model(opt)

    # Define optimization algorithm
    optimizer = torch.optim.Adam(model.net_g.parameters(), lr=learning_rate)

    #####################
    ##   4. TRAINING   ##
    #####################

    for e in range(epochs):
        losses = []
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            # Sample an image from dataset and make it a three channel (RGB) image
            small_images = batch[0].to(device)
            small_images = torch.cat((small_images, small_images, small_images), dim=1)
            big_images = batch[1].to(device)
            big_images = torch.cat((big_images, big_images, big_images), dim=1)


            # Get the model prediction for the noise
            pred = model.net_g(small_images)
            # Compare the predicted noise with the actual noise
            loss = F.l1_loss(pred, big_images)

            # Update the model parameters with the optimizer based on this loss
            loss.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        #######################
        ##   5. VALIDATION   ##
        #######################
        validation_losses = []
        for step, batch in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
            # Sample an image from dataset and make it a three channel (RGB) image
            small_images = batch[0].to(device)
            small_images = torch.cat((small_images, small_images, small_images), dim=1)
            big_images = batch[1].to(device)
            big_images = torch.cat((big_images, big_images, big_images), dim=1)


            # Get the model prediction for the noise
            pred = model.net_g(small_images)
            # Compare the predicted noise with the actual noise
            loss = F.l1_loss(pred, big_images)

            # Update the model parameters with the optimizer based on this loss
            validation_losses.append(loss.item())

        ####################
        ##   6. LOGGING   ##
        ####################

        # Use superresolution network
        output = model.net_g(small_images)
        lr_img = np.transpose(small_images[0,...].cpu().detach().numpy(), (1,2,0))
        output = np.transpose(output[0,...].cpu().detach().numpy(), (1,2,0))

        # Log images
        if wandb_enabled:
            lr_image = wandb.Image(lr_img, caption="Epoch " + str(e))
            wandb.log({"Low Res Image": lr_image})
            hr_image = wandb.Image(output, caption="Epoch " + str(e))
            wandb.log({"High Res Image": hr_image})
            wandb.log({"Training Loss" : sum(losses) / len(losses)})
            wandb.log({"Validation Loss" : sum(validation_losses) / len(validation_losses)})
        else:
            plt.imshow(lr_img, cmap='gray')
            plt.show()
            plt.imshow(output)
            plt.show()
        
        print("Epoch " + str(e) + ", Avg Loss: " + str(sum(losses) / len(losses)))
        print("Epoch " + str(e) + ", Avg Validation Loss: " + str(sum(validation_losses) / len(validation_losses)))

        # Save models and optimiser state every 10 epochs
        saving_interval = 50
        if (e+1) % saving_interval == 0:
            torch.save(model.net_g.state_dict(), 'HAT_ft_' + str(e) + '.pth')
            torch.save(optimizer.state_dict(), 'HAT_ft_optimizer.pth')
