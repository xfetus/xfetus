# Diffusion super resolution GAN 
The dataset used to train DSR-GAN model is the [FETAL_PLANES_DB dataset](https://zenodo.org/record/3904280).

## Training `DSRGAN`
The following bash scripts make use of [config files](../configs/) that need to be updated based on the location of your dataset.
* Generate data  
Please update [config_precompute_dataset_for_dsrgan.yaml](../configs/data/config_precompute_dataset_for_dsrgan.yaml) to update parameters according to your needs.
```
bash precompute_dataset_ddpm.bash
```
* Train model.   
Please update [config_train_baseline_ddpm.yaml](../configs/models/config_train_baseline_ddpm.yaml) to update parameters according to your needs.

```
cd scripts
bash train_ddpm.bash
# Provide API key for logging in to the wandb library at https://wandb.ai/xfetus
```

* Monitoring model training in wandb.  
When running `train_ddpm.bash` you can open your project at https://wandb.ai/
![fig](figures/wandb-board.png)

* Finishing training  
The terminal will show a message for logs and also data and models will be saved according to the setup path in config files.
```
wandb: 🚀 View run youthful-firefly-1 at: https://wandb.ai/xfetus/my-awesome-project/runs/ligypml3
wandb: Find logs at: wandb/run-20241231_181851-ligypml3/logs
```

## Huggingface

A pretrained version of the baseline can be downloaded from https://huggingface.co/harveymannering/xfetus-ddpm-v2.  
This page also contains training details and instructions for use.  

### Example Outputs
The figure below includes examples of both real and synthetic images. The following preprocessing and augmentation steps were applied to all training images:
Random Horizontal Flip
Random Rotation (±45°)
Resize to 128×128 using Bicubic Interpolation

The figure below includes examples of both real and synthetic images.  
<img width="608" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/6349716695ab8cce385f450e/RArVBPLLxPX_5rqSzXnp9.png">

### Training Loss
The baseline model was trained exclusively on images from the 'Voluson E6' machine. Training and validation losses are presented below. Checkpoints were saved every 50 epochs, and the best-performing checkpoint in terms of validation loss was found at epoch 250. The model provided here corresponds to the checkpoint from epoch 250.

<img width="608" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/6349716695ab8cce385f450e/XEZb34rdFYaeFckDMyCYm.png">

