#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler directive not a comment.

#########################FLAGS#################################

# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l mem=8G
#$ -l gpu=1

# only need 1 day to test this script.
#$ -l h_rt=72:00:00

# choose your preferred shell.
#$ -S /bin/bash

# merge STDOUT and STDERROR
#$ -j y

# make sure you give it a memorable name
#$ -N SimpleModel

# output directory for STDOUT file
#$ -o ~/runLog/

#########################/FLAGS#################################

#The code you want to run now goes here.

# print hostname and data for reference.
hostname
date

export LD_LIBRARY_PATH=/home/$USER/miniconda3/lib/:${LD_LIBRARY_PATH}

# optionally activate your conda env here:
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

#TODO use ve.yml
#source bin/activate
#pip install wandb
#pip install diffusers==0.12.1
#pip install monai
#pip install torchmetrics[image]
 
# cd somewhere where your code is.
# echo WANDB_LOGIN
# wandb login dc75d2e0e7638284f8029d745f2ac17c452e9de8
echo SCRIPT_START
python simple_model.py
echo SCRTIP_END

# print hostname and date for reference again
hostname
date

# cleanup
rm -r $COPYDIR

# give time for a clean exit.
sleep 10

date
