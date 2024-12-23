# Creating virtual environments

## Install mamba
Install [mamba](https://github.com/mxochicale/code/tree/main/mamba) 

## Create virtual environment
### simple env
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

  Summary:
  Install: 83 packages
  Total download: 479MB

...

mkl                                                209.3MB @ 538.3kB/s 1m:43.2s
cudatoolkit                                        872.0MB @   1.8MB/s 7m:54.6s
pytorch                                              1.5GB @   2.8MB/s 8m:58.7s

...





```

## Check dependencies

* Python package versions
```
$ cd $HOME/.../dependencies
$ mamba activate xfetusVE
$ python package_versions.py 



python: 3.11.4 | packaged by conda-forge | (main, Jun 10 2023, 18:08:17) [GCC 12.2.0]
torch: 2.0.0.post200
torchvision: 0.15.2a0+072ec57
torch cuda_is_available: True
torch cuda version: 11.2
torch cuda.device_count  1
PIL: 10.0.0

```

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

Timestamp                                 : Sat Dec 17 13:27:52 2022
Driver Version                            : 520.61.05
CUDA Version                              : 11.8

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : NVIDIA RTX A2000 8GB Laptop GPU
    Product Brand                         : NVIDIA RTX
    Product Architecture                  : Ampere

```


## conda-build
mamba install conda-build
conda skeleton pypi xfetus
