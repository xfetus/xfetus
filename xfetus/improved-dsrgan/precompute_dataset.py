import argparse
import os

from skimage import io
from skimage.transform import resize
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms

if __name__ == "__main__":


    # Command line aurgments - for script
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", help="File location of the fetal brain dataset", type=str)
    args = parser.parse_args()
    root_dir = args.dataset_path

    # Define augmentations for each image
    transform_operations = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(45),
    ])

    # Load dataset csv metadata
    #root_dir = '/content/FETAL_PLANES_ZENODO/'
    images_path = root_dir + 'Images/'
    csv_path = root_dir + 'FETAL_PLANES_DB_data.csv'
    csv_file = pd.read_csv(csv_path, sep=';')

    # Filter dataset
    e6_metadata = csv_file[csv_file['US_Machine'] == 'Voluson E6']

    # Define how many images will be in the results datasets
    train_dataset_size = 4000
    validation_dataset_size = 1000


    # Iterate though each plane
    planes = ['Other', 'Maternal cervix', 'Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax']
    for p in planes:
      # Filter by plane
      plane_metadata = e6_metadata[e6_metadata['Plane'] == p]

      # Get the training data
      train_metadata = plane_metadata[plane_metadata['Train '] == 1]
      print(len(train_metadata))

      # Define empty dataset as numpy array of zeros
      train_dataset = np.zeros((train_dataset_size, 64, 64), dtype = np.float32)

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
          small_image = cv2.resize(image[0,...].cpu().detach().numpy(), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

          # normalize image to be between -1 and 1
          small_image = small_image - 0.5

          # cast datatype to float32
          small_image = small_image.astype(np.float32)

          # Save the image into the dataset
          train_dataset[count,...] = small_image
          count += 1

      # Save training data
      np.save(p + ' train.npy', train_dataset)

      # Get the test data 
      validation_metadata = plane_metadata[plane_metadata['Train '] == 0]
      print(len(validation_metadata))

      # Define empty dataset as numpy array of zeros
      validation_dataset = np.zeros((validation_dataset_size, 64, 64), dtype = np.float32)

      # Create the validation dataset
      count = 0
      while count < validation_dataset_size:
        for index, row in validation_metadata.iterrows():
          if count >= validation_dataset_size:
            break
          # Validation set only contain even indexes, this effec tivly splits the test part of the
          # dataset 50/50 validation/test.
          elif index % 2 == 0:
            continue

          # Load image from dataset
          img_file_name = os.path.join(images_path, row['Image_name'] + '.png')
          image = io.imread(img_file_name)
          
          # Preprocess and augment the image
          image = transform_operations(image)
          small_image = cv2.resize(image[0,...].cpu().detach().numpy(), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

          # normalize image to be between -1 and 1
          small_image = small_image - 0.5

          # cast datatype to float32
          small_image = small_image.astype(np.float32)

          # Save the image into the dataset
          validation_dataset[count,...] = small_image
          count += 1
        print(count)
      np.save(p + ' validation.npy', validation_dataset)