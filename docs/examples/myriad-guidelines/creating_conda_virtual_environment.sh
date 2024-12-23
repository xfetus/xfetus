# Create a new python virtual environment
python -m venv .
# Activate the new environment
source bin/activate

# Install python packages in environment
#TODO use ve.yml
pip install diffusers==0.12.1 datasets accelerate wandb open-clip-torch
pip install monai
pip install opencv-python
pip install torch
pip install torchvision
pip install scikit-image
pip install tqdm
pip install wandb
pip install torchmetrics[image]