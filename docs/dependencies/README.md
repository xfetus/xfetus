# Creating virtual environments


## uv

###  Install uv: "An extremely fast Python package manager".
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create venv
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment:
# deactivate

#remove
#uv venv 2nd_env --python 3.13 #create with a diff python version
#rm -rf 2nd_env #to remove 2nd_env
```

### Activate VE and install lib
```
source .venv/bin/activate #To activate the virtual environment:
uv pip install --editable . # Install the package in editable mode
```

### lauch jupyter notebooks
``` 
source .venv/bin/activate #To activate the virtual environment:
#? export PYTHONPATH=.
jupyter notebook --browser=firefox
```


## conda/mamba

### Install mamba
Install [mamba](https://github.com/mxochicale/code/tree/main/mamba) 

## Create virtual environment
```
mamba update -n base mamba
mamba create -n xfetusVE python=3.8 pip -c conda-forge
mamba activate xfetusVE
```

### all dependencies mamba env 
* [ve.yml](ve.yml)

```
mamba update -n base mamba
mamba env create -f ve.yml
```

### build
mamba install conda-build
conda skeleton pypi xfetus


## Hardware 
* OS
```
$ hostnamectl

 Static hostname: --
       Icon name: computer-laptop
         Chassis: laptop
      Machine ID: --
         Boot ID: --
Operating System: Ubuntu 22.04.1 LTS              
          Kernel: Linux 5.15.0-56-generic
    Architecture: x86-64
 Hardware Vendor: --

```

* GPU
```
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                                 : Mon Dec 23 18:12:14 2024
Driver Version                            : 560.35.05
CUDA Version                              : 12.6

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA RTX A2000 8GB Laptop GPU
    Product Brand                         : NVIDIA RTX
    Product Architecture                  : Ampere

```
