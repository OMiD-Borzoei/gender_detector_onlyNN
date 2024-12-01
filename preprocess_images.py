from torch import flatten
import torchvision.transforms as transforms 

from config import input_width, input_height

def transform():
    # Specify any desired transformations  
    transform = transforms.Compose([  
        transforms.Resize((input_width, input_height)),  # Resize or any other transformations  
        transforms.ToTensor(),  # Convert PIL image to PyTorch Tensor  
        transforms.Lambda(lambda x: flatten(x)),  # Flatten to 1D tensor 
    ])
    
    return transform  