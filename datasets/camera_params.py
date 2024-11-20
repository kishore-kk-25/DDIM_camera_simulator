from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np,pandas as pd,os,sys
import glob

class CameraParameterDataset(Dataset):
    def __init__(self, data_path, csv_path, transform=None):
        """
        Args:
            data_path (str): Path to the directory containing images and CSV file.
            csv_filename (str): Name of the CSV file containing image paths and camera parameters.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.meta_data = pd.read_csv(self.csv_path)
        
        # Store column names
        self.columns = list(self.meta_data.columns)
        # print(f"CSV Columns: {self.columns}")
        
        # Total length
        # print(self.data_path)
        # print(glob.glob(self.data_path+'**/*.JPG'))
        self.length = len(glob.glob(self.data_path+'**/*.JPG'))
        # len(self.meta_data)
        print(f"Total samples: {self.length}")
        
        # Maximum values for normalization
        self.aperture_max = self.meta_data['aperture'].max()
        self.iso_max = self.meta_data['iso'].max()
        self.time_max = self.meta_data['time'].max()
        
        # print(f"Maximum Aperture: {self.aperture_max}")
        # print(f"Maximum ISO: {self.iso_max}")
        # print(f"Maximum Exposure Time: {self.time_max}")
        
        # Image paths
        self.all_paths = glob.glob(self.data_path+'**/*.JPG')#self.meta_data['path'].tolist()
        
        # Define image transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # You can add more transformations here
            ])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Retrieve the row from the CSV
        img_fpath=  self.all_paths[idx]
        fol_name = os.path.basename(os.path.dirname(img_fpath))
        img_name = os.path.basename(img_fpath)

        common_df_path ='ProcessedData/Nikon/AutoModenikon_train_512/'
        # print("path",common_df_path+fol_name+'/'+img_name)
        
        row = self.meta_data[self.meta_data['path']==common_df_path+fol_name+'/'+img_name].iloc[0]
        # print(row)
        # self.meta_data.iloc[idx]
        csv_fpath = row['path']
        # print("csv fpath",csv_fpath)
        # scene_name,img_name = os.path.basename(csv_fpath.split('/')[-2]),os.path.basename(csv_fpath.split('/')[-1])
        
        # Extract camera parameters
        iso = row['iso']
        exposure_time = row['time']
        aperture = row['aperture']
        
        # Normalize parameters
        iso_normalized = np.round(iso / self.iso_max, 3)
        exposure_normalized = np.round(exposure_time / self.time_max, 3)
        aperture_normalized = np.round(aperture / self.aperture_max, 3)
        
        # Create parameter vector
        param_vector = np.array([exposure_normalized, iso_normalized, aperture_normalized], dtype=np.float32)
        
        # Extract image file path
        csv_fpath = row['path']
        
        # Assuming 'path' is relative to data_path
        # img_fpath = os.path.join(self.data_path, scene_name,img_name)
        
        # Load image
        try:
            image = Image.open(img_fpath).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_fpath}: {e}")
            # Handle the error as needed, e.g., return a black image or skip
            image = Image.new("RGB", (256, 256))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert parameter vector to tensor
        param_vector = torch.tensor(param_vector, dtype=torch.float32)
        
        return image, param_vector


class CameraParameterDataset_csv(Dataset):
    def __init__(self, data_path, csv_path, transform=None):
        """
        Args:
            data_path (str): Path to the directory containing images and CSV file.
            csv_filename (str): Name of the CSV file containing image paths and camera parameters.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.meta_data = pd.read_csv(self.csv_path)
        
        # Store column names
        self.columns = list(self.meta_data.columns)
        # print(f"CSV Columns: {self.columns}")
        
        # Total length
        self.length = len(self.meta_data)
        print(f"Total samples: {self.length}")
        
        # Maximum values for normalization
        self.aperture_max = self.meta_data['aperture'].max()
        self.iso_max = self.meta_data['iso'].max()
        self.time_max = self.meta_data['time'].max()
        
        # print(f"Maximum Aperture: {self.aperture_max}")
        # print(f"Maximum ISO: {self.iso_max}")
        # print(f"Maximum Exposure Time: {self.time_max}")
        
        # Image paths
        self.all_paths = self.meta_data['path'].tolist()
        
        # Define image transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # You can add more transformations here
            ])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Retrieve the row from the CSV
        row = self.meta_data.iloc[idx]
        csv_fpath = row['path']
        scene_name,img_name = os.path.basename(csv_fpath.split('/')[-2]),os.path.basename(csv_fpath.split('/')[-1])
        
        # Extract camera parameters
        iso = row['iso']
        exposure_time = row['time']
        aperture = row['aperture']
        
        # Normalize parameters
        iso_normalized = np.round(iso / self.iso_max, 3)
        exposure_normalized = np.round(exposure_time / self.time_max, 3)
        aperture_normalized = np.round(aperture / self.aperture_max, 3)
        
        # Create parameter vector
        param_vector = np.array([exposure_normalized, iso_normalized, aperture_normalized], dtype=np.float32)
        
        # Extract image file path
        csv_fpath = row['path']
        
        # Assuming 'path' is relative to data_path
        img_fpath = os.path.join(self.data_path, scene_name,img_name)
        
        # Load image
        try:
            image = Image.open(img_fpath).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_fpath}: {e}")
            # Handle the error as needed, e.g., return a black image or skip
            image = Image.new("RGB", (256, 256))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert parameter vector to tensor
        param_vector = torch.tensor(param_vector, dtype=torch.float32)
        
        return image, param_vector
