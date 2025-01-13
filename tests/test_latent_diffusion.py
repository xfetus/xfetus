import os
from pathlib import Path

import yaml
from loguru import logger
import time
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from diffusers import DDIMScheduler, DDPMPipeline
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline

from xfetus.utils.datasets import AfricanFetalPlaneDataset, PrecomputedFetalPlaneDataset

file=Path().absolute()/"tests/config_test_latent_diffusion.yml"
with open(file, "r") as file:
   config_yaml = yaml.load(file, Loader=yaml.FullLoader)


def test_data_path():
   """
   Test data path
   pytest -vs tests/test_latent_diffusion.py::test_data_path
   TODO:
   https://github.com/ashleve/lightning-hydra-template/blob/main/tests/conftest.py
   """
   MODELS_PATH = os.path.join(str(Path.home()), config_yaml["MODELS_PATH"])
   DATASET_PATH = os.path.join(str(Path.home()), config_yaml["SPANISH_FETAL_PLANES_DATA_PATH"])      
   batch_size = config_yaml["model_hyperparameters"]["batch_size"]

   # define filenames for training data (saved as several numpy arrays)
   training_filenames = [
      'Fetal_abdomen_train.npy',
      'Fetal_brain_train.npy',
      'Fetal_femur_train.npy',
      'Fetal_thorax_train.npy',
      'Maternal_cervix_train.npy',
      'Other_train.npy',
   ]

   # define filenames for validation data (saved as several numpy arrays)
   validation_filenames = [
      'Fetal_abdomen_validation.npy',
      'Fetal_brain_validation.npy',
      'Fetal_femur_validation.npy',
      'Fetal_thorax_validation.npy',
      'Maternal_cervix_validation.npy',
      'Other_validation.npy',
   ]

   # Define augmentations for each image
   transform_operations = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(45),
   ])

   # create dataloader for training data
   train_dataset = PrecomputedFetalPlaneDataset(MODELS_PATH, training_filenames)
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

   # create dataloader for validation data
   validation_dataset = PrecomputedFetalPlaneDataset(MODELS_PATH, validation_filenames)
   validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

   logger.info(f" Length of trainset={len(train_dataset)} and valset={len(validation_dataset)}")
   logger.info(f" Length of trainloader={len(train_loader)} and valloader={len(validation_loader)}")


def test_train():
   """
   Test data path
   pytest -vs tests/test_latent_diffusion.py::test_train
   """
   # Define model variables
   DATASET_PATH = os.path.join(str(Path.home()), config_yaml["SPANISH_FETAL_PLANES_DATA_PATH"])         
   MODELS_PATH = os.path.join(str(Path.home()), config_yaml["MODELS_PATH"])
   MODEL_NAME = config_yaml["model"]["name"]
   MODEL_SAVEFLAG = config_yaml["model"]["save_flag"]
   

   wandb_enabled = config_yaml["wandb"]["enabled"]

   # Define hyperparameters
   image_size = config_yaml["model_hyperparameters"]["image_size"]
   batch_size = config_yaml["model_hyperparameters"]["batch_size"]
   epochs = config_yaml["model_hyperparameters"]["epochs"]
   learning_rate = config_yaml["model_hyperparameters"]["learning_rate"]
   grad_accumulation_steps = config_yaml["model_hyperparameters"]["grad_accumulation_steps"]

   # Are we using a GPU or CPU?
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   ####################
   ##   2. DATASET   ##
   ####################

   # define filenames for training data (saved as several numpy arrays)
   training_filenames = [
      'Fetal_abdomen_train.npy',
      'Fetal_brain_train.npy',
      'Fetal_femur_train.npy',
      'Fetal_thorax_train.npy',
      'Maternal_cervix_train.npy',
      'Other_train.npy',
   ]

   # define filenames for validation data (saved as several numpy arrays)
   validation_filenames = [
      'Fetal_abdomen_validation.npy',
      'Fetal_brain_validation.npy',
      'Fetal_femur_validation.npy',
      'Fetal_thorax_validation.npy',
      'Maternal_cervix_validation.npy',
      'Other_validation.npy',
   ]

   # Define augmentations for each image
   transform_operations = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(45),
   ])

   # create dataloader for training data
   train_dataset = PrecomputedFetalPlaneDataset(MODELS_PATH, training_filenames)
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

   # create dataloader for validation data
   validation_dataset = PrecomputedFetalPlaneDataset(MODELS_PATH, validation_filenames)
   validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

   ##############################
   ##   3. MODEL & OPTIMIZER   ##
   ##############################

   # Download pre trained diffusion model from huggingface
   diffusers_model_name = config_yaml["diffusers"]["model_name"]
   diffusers_vae_name = config_yaml["diffusers"]["vae_name"]
   
   # image_pipe = DDPMPipeline.from_pretrained(diffusers_model_name)
   vae = AutoencoderKL.from_pretrained(diffusers_vae_name)
   image_pipe = StableDiffusionPipeline.from_pretrained(diffusers_model_name, vae=vae)   

   image_pipe.to(device)

   # Add class conditioning to our UNet
   add_conditioning = True
   if add_conditioning:
      time_embed_dim = image_pipe.unet.time_embedding.linear_1.out_features
      total_classes = len(training_filenames)
      image_pipe.unet.config.class_embed_type = None
      image_pipe.unet.class_embedding = nn.Embedding(total_classes, time_embed_dim, device=device)

   # Define scheduler
   total_steps = config_yaml["model_optimiser"]["total_steps"]
   inference_steps = config_yaml["model_optimiser"]["inference_steps"]
   scheduler = DDIMScheduler(num_train_timesteps=total_steps, rescale_betas_zero_snr=True)
   scheduler.set_timesteps(inference_steps)
   scheduler.timesteps[0] = config_yaml["model_optimiser"]["scheduler_timesteps"]

   # Define optimization algorithm
   optimizer = torch.optim.Adam(image_pipe.unet.parameters(), lr=learning_rate)

   continues_training = config_yaml["model_optimiser"]["continues_training"]
   starting_epoch = config_yaml["model_optimiser"]["starting_epoch"] 
   lowest_validation_loss = config_yaml["model_optimiser"]["lowest_validation_loss"]  
   # if continues_training:
   #     image_pipe.unet.load_state_dict(torch.load('128xflawed_249.pth')) #Where to get 128xflawed_249.pth
   #     starting_epoch = 250
   #     optimizer.load_state_dict(torch.load('128x_optim_flawed.pth')) #Where to get 128x_optim_flawed.pth

   
   starttime = time.time()  # print(f'Starting training loop at {startt}')


   #####################
   ##   4. TRAINING   ##
   #####################
   logger.info("Training started")
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
         if add_conditioning:
               noise_pred = image_pipe.unet(noisy_images.float(), timesteps, class_labels, return_dict=False)[0]
         else:
               noise_pred = image_pipe.unet(noisy_images.float(), timesteps, return_dict=False)[0]

         # Compare the predicted noise with the actual noise
         loss = F.mse_loss(noise_pred, noise)

         # Update the model parameters with the optimizer based on this loss
         loss.backward(loss)
         losses.append(loss.item())

         # Gradient accumulation:
         if (step + 1) % grad_accumulation_steps == 0:
               optimizer.step()
               optimizer.zero_grad()

      # Output the average loss for a given epoch
      average_epoch_loss = sum(losses)/len(losses)

      #######################
      ##   5. VALIDATION   ##
      #######################

      validation_losses = []
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
         if add_conditioning:
               noise_pred = image_pipe.unet(noisy_images.float(), timesteps, class_labels, return_dict=False)[0]
         else:
               noise_pred = image_pipe.unet(noisy_images.float(), timesteps, return_dict=False)[0]

         # Compare the predicted noise with the actual noise
         loss = F.mse_loss(noise_pred, noise)

         # Save loss
         validation_losses.append(loss.item())

      # Output the average loss for a given epoch
      average_validation_loss = sum(validation_losses)/len(validation_losses)

      ####################
      ##   6. LOGGING   ##
      ####################

      # Log train/validation loss for this epoch
      logger.info(f"Epoch {e} average training loss: {average_epoch_loss}")
      logger.info(f"Epoch {e} average validation loss: {average_validation_loss}")
      if wandb_enabled:
         wandb.log({"Training Loss" : average_epoch_loss})
         wandb.log({"Validation Loss" : average_validation_loss})

      # Every few epochs do some extra logging
      logging_interval = 2
      if (e+1) % logging_interval == 0:

         # Generate a single random image via reverse diffusion process
         x = torch.randn(batch_size, 3, int(image_size), int(image_size)).to(device) # noise
         for i, t in tqdm(enumerate(scheduler.timesteps)):
               model_input = scheduler.scale_model_input(x, t)
               with torch.no_grad():
                  if add_conditioning:
                     # Conditiong on the 'Fetal brain' class (with index 1) because I am most familiar
                     # with what these images look like
                     class_label = torch.ones(1, dtype=torch.int64)
                     noise_pred = image_pipe.unet(model_input, t, class_label.to(device))["sample"]
                  else:
                     noise_pred = image_pipe.unet(model_input, t)["sample"]
               x = scheduler.step(noise_pred, t, x).prev_sample

         # Convert final image to numpy
         validation_img = np.transpose(x[0,...].detach().cpu().numpy(), (1,2,0))

         # Log outputs on www.wandb.ai
         if wandb_enabled:
               images = wandb.Image(validation_img, caption="Epoch " + str(e))
               wandb.log({"Diffusion Image": images})
               wandb.log({"Average pixel value": np.mean(x.detach().cpu().numpy())})
         # Log outputs normally (comment plt lines if you're not running this in a notebook)
         else:
               plt.imshow(validation_img)
               plt.show()
               logger.info("Average pixel value: " + str(np.mean(x.detach().cpu().numpy())))

      # Save model
      current_time_stamp = datetime.now().strftime("%d%b%Y_%H-%M-%S")
      MODEL_SAVE_PATHNAME = MODEL_NAME + '_' + datetime.now().strftime("%d%b%Y")
      MODEL_SAVE_PATH_NAME = MODELS_PATH+'/'+ MODEL_SAVE_PATHNAME

      if MODEL_SAVEFLAG and not os.path.exists(MODEL_SAVE_PATH_NAME):
         logger.info(f"Creating directory: {MODEL_SAVE_PATH_NAME}")
         os.makedirs(MODEL_SAVE_PATH_NAME)

      if MODEL_SAVEFLAG and lowest_validation_loss > average_validation_loss:
         model_path_name =  MODEL_SAVE_PATH_NAME + '/' + MODEL_NAME + '_' + str(e) + '_' + current_time_stamp + '.pth'
         torch.save(image_pipe.unet.state_dict(), model_path_name)
         lowest_validation_loss = average_validation_loss
         logger.info(f"Saved model: {model_path_name}")


   endtime = time.time()
   elapsedtime = endtime - starttime
   logger.info(f"Elapsed time for the training and validation loops: {elapsedtime/60} (mins)")
