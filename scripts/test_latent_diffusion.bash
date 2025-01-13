#!/bin/bash

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../
source .venv/bin/activate #To activate the virtual environment
# pytest -vs tests/test_latent_diffusion.py::test_data_path
pytest -vs tests/test_latent_diffusion.py::test_train
