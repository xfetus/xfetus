import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from skimage import io
from skimage.transform import resize
from torchvision import transforms

if __name__ == "__main__":
    """
    Script to create numpy arrays with train and validation datasets per label

    python src/xfetus/models/dsrgan/precompute_dataset.py -c configs/data/config_precompute_dataset_for_dsrgan.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Config filename including path", type=str)
    args = parser.parse_args()
    config_file = args.config_file
    config = OmegaConf.load(config_file)
    data_path = config.dataset.path
    models_path = config.dataset.models_path
    DATA_PATH = os.path.join(Path.home(), data_path)
    MODELS_PATH = os.path.join(Path.home(), models_path)

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # Load dataset csv metadata
    images_path = os.path.join(DATA_PATH, 'Images')
    csv_path = os.path.join(DATA_PATH, 'FETAL_PLANES_DB_data.csv')
    csv_file = pd.read_csv(csv_path, sep=';')
    image_size = config.dataset.image_size

    # Filter dataset
    e6_metadata = csv_file[csv_file['US_Machine'] == 'Voluson E6']

    # Define how many images will be in the results datasets
    train_dataset_size = config.dataset.train_dataset_size
    validation_dataset_size = config.dataset.validation_dataset_size

    # Define augmentations for each image
    transform_operations = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(45),
    ])

    # Iterate though each plane
    planes = ['Other', 'Maternal cervix', 'Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax']
    for p in planes:
      logger.info(f" Plane: {p}")
      # Filter by plane
      plane_metadata = e6_metadata[e6_metadata['Plane'] == p]

      # Get the training data
      train_metadata = plane_metadata[plane_metadata['Train '] == 1]
      logger.info(f" Length of train_metadata {len(train_metadata)}")

      # Define empty dataset as numpy array of zeros
      train_dataset = np.zeros((train_dataset_size, image_size, image_size), dtype = np.float32)

      # Precompute the training dataset
      count = 0
      while count < train_dataset_size:
        for index, row in train_metadata.iterrows():
          if count >= train_dataset_size:
            break

          # Load image from dataset
          img_file_name = os.path.join(images_path, row['Image_name'] + '.png')
          image = io.imread(img_file_name)
          # Preprocess and augment the image
          image = transform_operations(image)
          small_image = cv2.resize(image[0,...].cpu().detach().numpy(), dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

          # normalize image to be between -1 and 1
          small_image = small_image - 0.5

          # cast datatype to float32
          small_image = small_image.astype(np.float32)

          # Save the image into the dataset
          train_dataset[count,...] = small_image
          count += 1
        logger.info(f" Count in train dataset {count}")

      # Save training data
      os.chdir(MODELS_PATH)
      np.save(p.replace(" ", "_") + '_train.npy', train_dataset)

      # Get the test data
      validation_metadata = plane_metadata[plane_metadata['Train '] == 0]
      logger.info(f" Length of validation_metadata {len(validation_metadata)}")

      # Define empty dataset as numpy array of zeros
      validation_dataset = np.zeros((validation_dataset_size, image_size, image_size), dtype = np.float32)

      # Create the validation dataset
      count = 0
      while count < validation_dataset_size:
        for index, row in validation_metadata.iterrows():
          if count >= validation_dataset_size:
            break
          # Validation set only contain even indexes, this effect tivly splits the test part of the
          # dataset 50/50 validation/test.
          elif index % 2 == 0:
            continue

          # Load image from dataset
          img_file_name = os.path.join(images_path, row['Image_name'] + '.png')
          image = io.imread(img_file_name)

          # Preprocess and augment the image
          image = transform_operations(image)
          small_image = cv2.resize(image[0,...].cpu().detach().numpy(), dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

          # normalize image to be between -1 and 1
          small_image = small_image - 0.5

          # cast datatype to float32
          small_image = small_image.astype(np.float32)

          # Save the image into the dataset
          validation_dataset[count,...] = small_image
          count += 1
        logger.info(f" Count in validation dataset {count}")


      # Save validation data
      os.chdir(MODELS_PATH)
      np.save(p.replace(" ", "_") + '_validation.npy', validation_dataset)
