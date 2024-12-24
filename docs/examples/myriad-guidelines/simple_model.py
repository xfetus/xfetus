from diffusers import DDIMScheduler, DDPMPipeline
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
import os
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import monai.transforms as mt
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers import DDIMScheduler
from skimage import io
import wandb
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance

class FetalPlaneDataset(Dataset):
    """Fetal Plane dataset."""

    def __init__(self, root_dir, ref, 
                 plane, 
                 brain_plane=None, 
                 us_machine=None, 
                 operator_number=None, 
                 transform=None
                ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            plane: 'Fetal brain'; 'Fetal thorax'; 'Maternal cervix'; 'Fetal femur'; 'Fetal thorax'; 'Other'
            brain_plane: 'Trans-ventricular'; 'Trans-thalamic'; 'Trans-cerebellum'
            us_machine: 'Voluson E6';'Voluson S10'
            operator_number: 'Op. 1'; 'Op. 2'; 'Op. 3';'Other'
            
        return image
        """
        self.root_dir = root_dir
        self.ref = pd.read_csv(ref, sep=';')
        self.ref = self.ref[self.ref['Plane'] == plane]
        if plane == 'Fetal brain':
            print(brain_plane)
            self.ref = self.ref[self.ref['Brain_plane'] == brain_plane]
        if us_machine is not None:
            self.ref = self.ref[self.ref['US_Machine'] == us_machine]
        if operator_number is not None:
            self.ref = self.ref[self.ref['Operator'] == operator_number]

        dataset_size = 256
        self.ref = self.ref[:dataset_size]
        self.transform = transform

    def __len__(self):
        return len(self.ref)

    def __getitem__(self, idx):
        
        # Load image
        img_name = os.path.join(self.root_dir,
                                self.ref.iloc[idx, 0] + '.png')        
        image = io.imread(img_name)
        
        # Apply transforms (augment) to images
        if self.transform:
            image = self.transform(image)

        return image



if __name__ == "__main__":

    # Weights and biases (wandb) is used for loggging outputs and is optional
    wandb_enabled = False
    if wandb_enabled:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",
            
            # track hyperparameters and run metadata
            config={
                "architecture": "CNN",
                "dataset": "Fetal Plane dataset",
            }
        )

        wandb.log({"Started" : 1})

    HOME_PATH = os.path.expanduser(f'~')
    USERNAME = os.path.split(HOME_PATH)[1]

    ###TOCHANGE FOR YOUR DATAPATH
    FULL_DATA_REPO_PATH = 'Scratch/FETAL_PLANES_ZENODO/'
    CSV_FILENAME_CSV = 'FETAL_PLANES_DB_data.csv'
    dataroot = FULL_DATA_REPO_PATH + "Images"
    ref = FULL_DATA_REPO_PATH + CSV_FILENAME_CSV

    # Are we using CPU or GPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Printing Versions and paths
    print(f'FULL_DATA_REPO_PATH: {FULL_DATA_REPO_PATH}' )
    print(f'Device: {device}')

    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Plane
    plane = 'Fetal brain'
    # Operator number 
    operator_number = None
    # Brain plan and ultrasound device
    brain_plane = 'Trans-ventricular'; us_machine = 'Voluson E6' ###len: 408 
    # Image dimensions (image will be square)
    image_size = 128
    # Number of workers for dataloader
    workers = 8
    # Batch size during training
    batch_size = 4

    # Define the training dataloader 
    transform_operations=transforms.Compose([
                            transforms.ToTensor(),
                            mt.RandRotate(range_x=0.1, prob=0.5),
                            mt.RandZoom(prob=0.5, min_zoom=1, max_zoom=1.1),
                            mt.RandFlip(prob=0.5, spatial_axis=1),
                            mt.Resize([image_size, image_size]),
                            transforms.Resize([image_size, image_size]),
                            transforms.Normalize(0.5, 0.5), #mean=0.5, std=0.5 
                            ])
    train_set = FetalPlaneDataset(root_dir=dataroot,
                                ref=ref,
                                plane=plane,
                                brain_plane=brain_plane,
                                us_machine=us_machine,
                                operator_number=operator_number,
                                transform=transform_operations)
    train_set = DataLoader(train_set, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=workers)
    
    # Download pre trained models from HuggingFace
    print("Downloading Models")
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=40)

    # Hyperparameters
    #num_epochs = 20001
    num_epochs = 2
    lr = 1e-4  
    grad_accumulation_steps = 8

    # Optimization algorithm
    optimizer = torch.optim.Adam(image_pipe.unet.parameters(), lr=lr)

    # Array for score history
    losses = []
    
    # Static noise
    x_noise = torch.randn(batch_size, 3, image_size, image_size).to(device) # noise
    x_noise = x_noise.to(device)

    # Define object to calculate the FID score
    fid = FrechetInceptionDistance(feature=192).to('cuda')

    print("Starting Training")
    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_set), total=len(train_set)):
            clean_images = batch.to(device)
            clean_images = torch.cat((clean_images, clean_images, clean_images), dim=1)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)
            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]
            # Compare the prediction with the actual noise:
            loss = F.mse_loss(
                noise_pred, noise
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

            # Store for later plotting
            losses.append(loss.item())

            # Update the model parameters with the optimizer based on this loss
            loss.backward(loss)

            # Gradient accumulation:
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            

        print(
            f"Epoch {epoch} average loss: {sum(losses[-len(train_set):])/len(train_set)}"
        )
        wandb.log({"Loss" : sum(losses[-len(train_set):])/len(train_set)})

        # Every 100th epoch output FID and an example synthetic image
        if epoch % 100 == 0: 
          
          # Generate a new images x via diffusion process
          x = torch.clone(x_noise)
          for i, t in tqdm(enumerate(scheduler.timesteps)):
            model_input = scheduler.scale_model_input(x, t)
            with torch.no_grad():
                  noise_pred = image_pipe.unet(model_input, t)["sample"]
            x = scheduler.step(noise_pred, t, x).prev_sample
          
          # Log images in weights and biases
          if wandb_enabled:
            images = wandb.Image(np.transpose(x[0,...].detach().cpu().numpy(), (1,2,0)), caption="Top: Output, Bottom: Input")
            wandb.log({"Diffusion": images})
          
          # Calculate FID score for current images
          clean_images = (clean_images + 1) * 127.5
          fid.update(clean_images.byte(), real=True)
          x = (x + 1) * 127.5
          fid.update(x.byte(), real=False)          
          current_fid = fid.compute().item()
          fid.reset()

          # Output the FID score
          print("FID", current_fid)
          if wandb_enabled:
            wandb.log({"FID": current_fid}) 

        # Save optimiser and network checkpoints 
        if epoch % 1000 == 0: 
          torch.save(image_pipe.unet.state_dict(), '128xfetal_half_noblur_tv' + str(epoch) + '.pth')
          torch.save(optimizer.state_dict(), '128x_optim_fetal_half_noblur_tv' + str(epoch) + '.pth')
