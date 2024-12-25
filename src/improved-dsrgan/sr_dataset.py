import os

from torch.utils.data import Dataset
import pandas as pd
from skimage import io
from skimage.transform import resize
import cv2
import numpy as np

class FetalPlaneDataset(Dataset):
    """Fetal Plane dataset."""

    def __init__(self, root_dir, csv_file, plane, 
                 brain_plane=None, 
                 us_machine=None, 
                 operator_number=None, 
                 transform=None,
                 train=None,
                 validation=False,
                 size=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            plane: 'Fetal brain'; 'Fetal thorax'; 'Maternal cervix'; 'Fetal femur'; 'Fetal thorax'; 'Other'
            brain_plane: 'Trans-ventricular'; 'Trans-thalamic'; 'Trans-cerebellum'
            us_machine: 'Voluson E6';'Voluson S10'
            operator_number: 'Op. 1'; 'Op. 2'; 'Op. 3';'Other'
            transform: Transformation that will be applied to all images
            size: dimension of the image (one number because images will be square)

            
        return image
        """

        # Select which images in the dataset to use
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file, sep=';')
        if plane is not None:
            self.csv_file = self.csv_file[self.csv_file['Plane'] == plane]
        if plane == 'Fetal brain' and brain_plane is not None:
            self.csv_file = self.csv_file[self.csv_file['Brain_plane'] == brain_plane]
        if us_machine is not None:
            self.csv_file = self.csv_file[self.csv_file['US_Machine'] == us_machine]
        if operator_number is not None:
            self.csv_file = self.csv_file[self.csv_file['Operator'] == operator_number]
        if train is not None:
            self.csv_file = self.csv_file[self.csv_file['Train '] == train]
        if validation == True:
            self.csv_file = self.csv_file[self.csv_file.index % 2 == True]


        # Save image properties
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        # Load the image from file
        img_file_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx, 0] + '.png')
        image = io.imread(img_file_name)
        
        # Preprocess and augment the image
        if self.transform:
            image = self.transform(image)

        # Resize the image
        image = image - 0.5
        image = resize(image, (image.shape[0], self.size, self.size))
        small_image = cv2.resize(image[0], dsize=(int(self.size/4), int(self.size/4)), interpolation=cv2.INTER_CUBIC)
        small_image = np.expand_dims(small_image, axis=0)

        return small_image, image
