from random import shuffle
from PIL import Image 
import os 
import torchvision.transforms as transforms  
from torch.utils.data import Dataset, DataLoader  

from config import test_portion, detail_view

class UTKFaceDataset(Dataset):  
    def __init__(self, directory, transform=None, train_set = False, test_set = False):  
        self.directory = directory  
        self.transform = transform  
        self.image_files = os.listdir(directory) 

        shuffle(self.image_files)        
        if detail_view:
            print("Total Fetched Image: ", len(self.image_files))
            print("Train Images: ", len(self.image_files[int(len(self.image_files)*test_portion):]))
            print("Test Images: ", len(self.image_files[:int(len(self.image_files)*test_portion)]))
            
        if train_set and not test_set:
            self.image_files = self.image_files[int(len(self.image_files)*test_portion):]
        
        if test_set and not train_set:
            self.image_files = self.image_files[:int(len(self.image_files)*test_portion)]

    def __len__(self):  
        return len(self.image_files)  

    def __getitem__(self, idx):  
        img_name = os.path.join(self.directory, self.image_files[idx])  
        image = Image.open(img_name)  

        if self.transform:  
            image = self.transform(image)  

        label = int(self.image_files[idx].split('_')[1]) # 1 = Woman, 0 = Man
        return image, label  # You can return labels or any other info as needed  

