# Improve DSR-GAN

The dataset used to train DSR-GAN model is the [FETAL_PLANES_DB dataset](https://zenodo.org/record/3904280).

A pretrained version of the baseline can be downloaded from https://huggingface.co/harveymannering/xfetus-ddpm-v2.  
This page also contains training details and instructions for use.  

## Example Outputs
The figure below includes examples of both real and synthetic images. The following preprocessing and augmentation steps were applied to all training images:
Random Horizontal Flip
Random Rotation (±45°)
Resize to 128×128 using Bicubic Interpolation

The figure below includes examples of both real and synthetic images.  
<img width="608" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/6349716695ab8cce385f450e/RArVBPLLxPX_5rqSzXnp9.png">


## Training Loss
The baseline model was trained exclusively on images from the 'Voluson E6' machine. Training and validation losses are presented below. Checkpoints were saved every 50 epochs, and the best-performing checkpoint in terms of validation loss was found at epoch 250. The model provided here corresponds to the checkpoint from epoch 250.

<img width="608" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/6349716695ab8cce385f450e/XEZb34rdFYaeFckDMyCYm.png">

