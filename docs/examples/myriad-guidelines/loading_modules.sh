export LD_LIBRARY_PATH=/home/$USER/miniconda3/lib/:${LD_LIBRARY_PATH}

# import all software thats already installed on myriad
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu