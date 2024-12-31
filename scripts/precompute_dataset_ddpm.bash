#!/bin/bash

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../
source .venv/bin/activate #To activate the virtual environment
python src/xfetus/models/dsrgan/precompute_dataset.py -c configs/data/config_precompute_dataset_for_dsrgan.yaml
