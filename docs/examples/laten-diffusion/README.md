# Laten Diffusion 

## Download datasets 
* african-fetal-us-2023
```
mkdir -p ~/datasets/african-fetal-us-2023 && cd ~/datasets/african-fetal-us-2023
wget https://zenodo.org/records/7540448/files/Zenodo_dataset.tar.xz
tar xf Zenodo_dataset.tar.xz #[41MB]
rm Zenodo_dataset.tar.xz
#mv all images to root data path
```
* FETAL_PLANES_ZENODO
```
mkdir -p ~/datasets/FETAL_PLANES_DB_2020 && cd ~/datasets/FETAL_PLANES_DB_2020
wget https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip #
unzip FETAL_PLANES_ZENODO.zip
rm FETAL_PLANES_ZENODO.zip
```
See further details [here](/data/datasets/)

## Testing train model pipeline
The following bash scripts make use of [config files](../configs/) that need to be updated based on the location of your dataset.

* Generate data. Please update [config_precompute_dataset_for_dsrgan.yaml](../configs/data/config_precompute_dataset_for_dsrgan.yaml) to update variables in [precompute_datasets.py](../../../src/xfetus/models/dsrgan/precompute_dataset.py) according to your needs.
```
bash scripts/precompute_dataset_ddpm.bash using   
```

* Developing and testing train pipeline [config_test_latent_diffusion.yml](../../../tests/config_test_latent_diffusion.yml).
```
bash scripts/test_latent_diffusion.bash
```

 * File outputs will look like:
 ```
~/datasets/FETAL_PLANES_DB_2020/models/latentdiffusion$ tree -h
[4.0K]  .
├── [4.0K]  128x_baseline_13Jan2025
│   ├── [434M]  128x_baseline_0_13Jan2025_06-17-37.pth
│   └── [434M]  128x_baseline_0_13Jan2025_06-20-45.pth
├── [1.2M]  Fetal_abdomen_train.npy
├── [320K]  Fetal_abdomen_validation.npy
├── [1.2M]  Fetal_brain_train.npy
├── [320K]  Fetal_brain_validation.npy
├── [1.2M]  Fetal_femur_train.npy
├── [320K]  Fetal_femur_validation.npy
├── [1.2M]  Fetal_thorax_train.npy
├── [320K]  Fetal_thorax_validation.npy
├── [1.2M]  Maternal_cervix_train.npy
├── [320K]  Maternal_cervix_validation.npy
├── [1.2M]  Other_train.npy
└── [320K]  Other_validation.npy

1 directory, 14 files
 ```