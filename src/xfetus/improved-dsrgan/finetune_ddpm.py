from os import path as osp

from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import wandb
from diffusers import DDIMScheduler, DDPMPipeline
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from haar_pytorch import HaarForward, HaarInverse

from loss import loss_img, loss_hf, loss_hfmse
from ddpm_dataset import PrecomputedFetalPlaneDataset

if __name__ == "__main__":

    ##################
    ##   1. SETUP   ##
    ##################

    # Command line aurgments - for script
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="File location of the fetal brain dataset", type=str)
    parser.add_argument("-w", "--wandb_enabled", help="Enable weights and bias logging", type=bool)
    parser.add_argument("-c", "--ddpm_checkpoint_path", default=None,  help="File location for the pre-trained DDPM model", type=str)
    parser.add_argument("-l", "--ddpm_pa_loss", default=None,  help="Use DDPM-PA loss (as opposed to standard DDPM loss)", type=bool)
    args = parser.parse_args()
    wandb_enabled = args.wandb_enabled
    dataset_path = args.dataset_path
    ddpm_checkpoint_path = args.ddpm_checkpoint_path
    ddpm_pa_loss = args.ddpm_pa_loss
    # Command line aurgments - for google colab
    '''wandb_enabled = False
    dataset_path = "/content/gdrive/MyDrive/finetuning_data/"
    ddpm_checkpoint_path = "/content/gdrive/MyDrive/128xflawed_349.pth"
    ddpm_pa_loss = True'''

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

    # Define hyperparameters
    image_size = 128
    batch_size = 4
    epochs = 100
    learning_rate = 1e-4
    grad_accumulation_steps = 8

    # Are we using a GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ####################
    ##   2. DATASET   ##
    ####################
    
    # define filenames for training data (saved as several numpy arrays)
    training_filenames = [
        'Fetal abdomen train.npy',
        'Fetal brain train.npy',
        'Fetal femur train.npy',
        'Fetal thorax train.npy'
    ]

    # define filenames for validation data (saved as several numpy arrays)
    validation_filenames = [
        'Fetal abdomen validation.npy',
        'Fetal brain validation.npy',
        'Fetal femur validation.npy',
        'Fetal thorax validation.npy'
    ]

    # Define augmentations for each image
    transform_operations = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(45),
    ])

    # create dataloader for training data
    train_dataset = PrecomputedFetalPlaneDataset(dataset_path + 'train/', training_filenames)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # create dataloader for validation data
    validation_dataset = PrecomputedFetalPlaneDataset(dataset_path + 'validation/', validation_filenames)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    ##############################
    ##   3. MODEL & OPTIMIZER   ##
    ##############################

    # Download pre trained diffusion model from huggingface
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # Add class conditioning to our UNet
    add_conditioning = True
    if add_conditioning:
        time_embed_dim = image_pipe.unet.time_embedding.linear_1.out_features
        total_classes = 6 # baseline classes are : abdomen, brain, femur, thorax, cervix, and other
        image_pipe.unet.config.class_embed_type = None
        image_pipe.unet.class_embedding = nn.Embedding(total_classes, time_embed_dim, device=device)

    # Define optimization algorithm
    optimizer = torch.optim.Adam(image_pipe.unet.parameters(), lr=learning_rate)

    # Load checkpoint
    continues_training = True
    starting_epoch = 0
    if continues_training:
        image_pipe.unet.load_state_dict(torch.load(ddpm_checkpoint_path))

    # Copy the pretrained model and freeze parameters
    if ddpm_pa_loss:
        image_pipe_baseline = copy.deepcopy(image_pipe)
        for param in image_pipe_baseline.unet.parameters():
            param.requires_grad = False
    
    lowest_validation_loss = 10

    #####################
    ##   4. TRAINING   ##
    #####################
    print("Training started")
    for e in range(starting_epoch, epochs):
        losses = []
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Sample an image from dataset and make it a three channel (RGB) image
            clean_images, class_labels = batch
            clean_images = torch.unsqueeze(clean_images, 1) 
            clean_images = torch.cat((clean_images, clean_images, clean_images), dim=1)

            # Move data to whatever device we are using
            clean_images = clean_images.to(device)
            class_labels = class_labels.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            # Forward diffusion process (Add noise to the clean images according to the noise magnitude at each timestep)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(noisy_images.float(), timesteps, class_labels, return_dict=False)[0]

            # Compare the predicted noise with the actual noise
            loss = F.mse_loss(noise_pred, noise)

            if ddpm_pa_loss:
                # Scalar weightings for loss function terms
                lambdas = [0.001, 0.0, 0.5, 0.05]

                # Get the model prediction for the noise
                noise_pred_src = image_pipe_baseline.unet(noisy_images.float(), timesteps, class_labels, return_dict=False)[0]
              
                # Limg
                loss += lambdas[1] * loss_img(
                    noise_pred, 
                    noise_pred_src,
                    noisy_images.float(), 
                    image_pipe.scheduler.alphas_cumprod.to(device), 
                    timesteps
                )

                # Lhf
                loss += lambdas[2] * loss_hf(
                    noise_pred, 
                    noise_pred_src,
                    noisy_images.float(), 
                    image_pipe.scheduler.alphas_cumprod.to(device), 
                    timesteps
                )

                # Lhfmse
                loss += lambdas[3] * loss_hfmse(
                    noise_pred, 
                    noise_pred_src,
                    noisy_images.float(), 
                    image_pipe.scheduler.alphas_cumprod.to(device), 
                    timesteps
                )

            # Update the model parameters with the optimizer based on this loss
            loss.backward(loss)
            losses.append(loss.item())

            # Gradient accumulation:
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Output the average loss for a given epoch
        average_epoch_loss = sum(losses) / len(losses)

        #######################
        ##   5. VALIDATION   ##
        #######################

        validation_losses = []
        validation_losses_simple = []
        for step, batch in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            # Sample an image from dataset and make it a three channel (RGB) image
            clean_images, class_labels = batch
            clean_images = torch.unsqueeze(clean_images, 1) 
            clean_images = torch.cat((clean_images, clean_images, clean_images), dim=1)

            # Move data to whatever device we are using
            clean_images = clean_images.to(device)
            class_labels = class_labels.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            # Forward diffusion process (Add noise to the clean images according to the noise magnitude at each timestep)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(noisy_images.float(), timesteps, class_labels, return_dict=False)[0]

            # Compare the predicted noise with the actual noise
            loss = F.mse_loss(noise_pred, noise)
            validation_losses_simple.append(loss.item())
            if ddpm_pa_loss:

                # Get the model prediction for the noise
                noise_pred_src = image_pipe_baseline.unet(noisy_images.float(), timesteps, class_labels, return_dict=False)[0]
              
                # Limg
                loss += lambdas[1] * loss_img(
                    noise_pred, 
                    noise_pred_src,
                    noisy_images.float(), 
                    image_pipe.scheduler.alphas_cumprod.to(device), 
                    timesteps
                )

                # Lhf
                loss += lambdas[2] * loss_hf(
                    noise_pred, 
                    noise_pred_src,
                    noisy_images.float(), 
                    image_pipe.scheduler.alphas_cumprod.to(device), 
                    timesteps
                )

                # Lhfmse
                loss += lambdas[3] * loss_hfmse(
                    noise_pred, 
                    noise_pred_src,
                    noisy_images.float(), 
                    image_pipe.scheduler.alphas_cumprod.to(device), 
                    timesteps
                )

            # Save loss
            validation_losses.append(loss.item())

        # Output the average loss for a given epoch
        average_validation_loss = sum(validation_losses)/len(validation_losses)
        average_validation_loss_simple =  sum(validation_losses_simple)/len(validation_losses_simple)

        ####################
        ##   6. LOGGING   ##
        ####################

        # Log train/validation loss for this epoch
        print(f"Epoch {e} average training loss: {average_epoch_loss}")
        print(f"Epoch {e} average validation loss: {average_validation_loss}")
        if wandb_enabled:
            wandb.log({"Training Loss" : average_epoch_loss})
            wandb.log({"Validation Loss" : average_validation_loss})
            wandb.log({"Simple Validation Loss" : average_validation_loss_simple})

        # Every few epochs do some extra logging
        logging_interval = 2
        if (e+1) % logging_interval == 0:

            # Generate a single random image via reverse diffusion process
            x = torch.randn(batch_size, 3, int(image_size), int(image_size)).to(device) # noise
            for i, t in tqdm(enumerate(image_pipe.scheduler.timesteps)):
                model_input = image_pipe.scheduler.scale_model_input(x, t)
                with torch.no_grad():
                    # Conditiong on the 'Fetal brain' class (with index 1) because I am most familar 
                    # with what these images look like
                    class_label = torch.ones(1, dtype=torch.int64)
                    noise_pred = image_pipe.unet(model_input, t, class_label.to(device))["sample"]

                x = image_pipe.scheduler.step(noise_pred, t, x).prev_sample
                
            # Convert final image to numpy
            validation_img = np.transpose(x[0,...].detach().cpu().numpy(), (1,2,0))
            
            # Log outputs on www.wandb.ai
            if wandb_enabled:
                images = wandb.Image(validation_img, caption="Epoch " + str(e))
                wandb.log({"Diffusion Image": images})
                wandb.log({"Average pixel value": np.mean(x.detach().cpu().numpy())})
            # Log outputs normally (comment plt lines if you're not running this in a notebook)
            else:
                plt.imshow(validation_img + 0.5)
                plt.show()
                print("Average pixel value: " + str(np.mean(x.detach().cpu().numpy())))


        # Save models and optimiser state every 1000 epochs
        if lowest_validation_loss > average_validation_loss:
            torch.save(image_pipe.unet.state_dict(), '128xft_ddpm-pa.pth')
            lowest_validation_loss = average_validation_loss
    
