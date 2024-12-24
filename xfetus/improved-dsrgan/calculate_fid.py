from torchmetrics.image.fid import FrechetInceptionDistance
import torch
from diffusers import DDPMPipeline
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    ##################
    ##   1. SETUP   ##
    ##################

    # Command line aurgments - for script
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="File location of the fetal brain dataset", type=str)
    parser.add_argument("-c", "--ddpm_checkpoint_path", default=None,  help="File location for the pre-trained DDPM model", type=str)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    ddpm_checkpoint_path = args.ddpm_checkpoint_path

    # Are we using a GPU or CPU?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    starting_epoch = 0
    image_pipe.unet.load_state_dict(torch.load(ddpm_checkpoint_path))

    # Setup FID metric
    fid = FrechetInceptionDistance(feature=192).to('cuda')

    # Define hyperparameters
    image_size = 128
    batch_size = 4
    total_images = 2048

    # Load the real images
    print("Generate real images")
    real_images = np.load(dataset_path)
    real_images = torch.from_numpy(real_images[:total_images,...])
    real_images = torch.stack([real_images, real_images, real_images], dim=1)
    real_images = torch.clip(real_images, -0.5, 0.5)
    real_tensor = ((real_images + 0.5) * 255).byte()
    
    # Generate fake images
    print("Generate fake images") 
    fake_images = np.zeros((total_images, 3, int(image_size), int(image_size)))
    for img_idx in range(0, total_images, batch_size):
        x = torch.randn(batch_size, 3, int(image_size), int(image_size)).to(device) # noise
        for i, t in tqdm(enumerate(image_pipe.scheduler.timesteps)):
            model_input = image_pipe.scheduler.scale_model_input(x, t)
            with torch.no_grad():
                if add_conditioning:
                    # Conditiong on the 'Fetal brain' class (with index 1) because I am most familar 
                    # with what these images look like
                    class_label = torch.ones(1, dtype=torch.int64)
                    noise_pred = image_pipe.unet(model_input, t, class_label.to(device))["sample"]
                else:
                    noise_pred = image_pipe.unet(model_input, t)["sample"]
            x = image_pipe.scheduler.step(noise_pred, t, x).prev_sample
        fake_images[img_idx:img_idx+batch_size] = x.cpu().detach().numpy()
    fake_tensor = torch.from_numpy(np.mean(fake_images, axis=1))
    fake_tensor = torch.stack([fake_tensor, fake_tensor, fake_tensor], dim=1)
    fake_tensor = torch.clip(fake_tensor, -0.5, 0.5)
    fake_tensor = ((fake_tensor + 0.5) * 255).byte()

    #  Calcuate the FID score 
    fid.update(real_tensor.to(device), real=True)        
    fid.update(fake_tensor.to(device), real=False)
    current_fid = fid.compute().item()
    print("FID Score: ", current_fid)
    print(dataset_path)
    fid.reset()

    # Save images
    np.save('fake_images.npy', fake_images)



