import os 
import torch  
from torch.utils.data import DataLoader 

from data_loader import UTKFaceDataset
from models import SimpleNN_2HiddenLayer as SimpleNN
from preprocess_images import transform
from train import train 
from utils import set_seed, set_device
from losses import CustomBCEWithLogitsLoss
from config import lr, batch_size, result_path 

if __name__ == "__main__":
    os.makedirs(result_path, exist_ok=True)
    
    SEED = 99243020
    set_seed(SEED)
    
    DEVICE = set_device()
    
    image_folder = "UTKFace"
    train_dataset = UTKFaceDataset(directory=image_folder, transform=transform(), train_set=True)  
    test_dataset = UTKFaceDataset(directory=image_folder, transform=transform(), test_set=True)  
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

    model = SimpleNN()
    model.to(DEVICE)
    
    criterion = CustomBCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  
    
    train(model, criterion, optimizer, train_data_loader,test_data_loader, DEVICE)

