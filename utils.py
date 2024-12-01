import torch 
from torch import nn 
import random 
import numpy as np 
 

def set_seed(seed=None, seed_torch=True):
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def set_device() -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    return device 
