#!/bin/bash

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../
source .venv/bin/activate #To activate the virtual environment
python src/xfetus/models/dsrgan/train_baseline_ddpm.py -c configs/models/config_train_baseline_ddpm.yaml
