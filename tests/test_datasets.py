import os
from pathlib import Path

import yaml
from loguru import logger
from torchvision import transforms

from xfetus.utils.datasets import AfricanFetalPlaneDataset

file=Path().absolute()/"tests/config_test.yml"
with open(file, "r") as file:
   config_yaml = yaml.load(file, Loader=yaml.FullLoader)


def test_data_path():
    """
    Test data path
    pytest -vs tests/test_datasets.py::test_data_path

    TODO:
    https://github.com/ashleve/lightning-hydra-template/blob/main/tests/conftest.py
    """
    DATASET_PATH = os.path.join(str(Path.home()), config_yaml["AFRICAN_DATA_PATH"])

    DATABASE_CSV=DATASET_PATH+"/African_planes_database.csv"
    image_size = 224
    transform_operations=transforms.Compose([
                             #mt.RandRotate(range_x=0.1, prob=0.5),
                             #mt.RandZoom(prob=0.5, min_zoom=1, max_zoom=1.1),
                             #mt.Resize([image_size, image_size]),
                             transforms.Grayscale(num_output_channels=3),#mean=0.5, std=0.5
                             transforms.ToTensor(),
                             transforms.Resize([image_size, image_size], antialias=True),
                             transforms.Normalize((0.5), (0.5)),
                             ])

    african_train_dataset = AfricanFetalPlaneDataset(DATASET_PATH,
                                                     DATABASE_CSV,
                                               transform=transform_operations,
                                               return_labels=True,
                                               split_type="csv",
                                               split="train")
    african_val_dataset = AfricanFetalPlaneDataset(DATASET_PATH,
                                               DATABASE_CSV,
                                               transform=transform_operations,
                                               return_labels=True,
                                               split_type="csv",
                                               split="valid")

    logger.info(f" Length of trainset={len(african_train_dataset)} and valset={len(african_val_dataset)}")

    assert len(african_train_dataset) == 217
    assert len(african_val_dataset) == 233
