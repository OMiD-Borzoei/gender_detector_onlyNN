import torch 
from torch import nn, sigmoid 
from actiavtion_functions import CustomReLU

from config import input_dimension, first_hidden_layer_neurons, second_hidden_layer_neurons, third_hidden_layer_neurons, fourth_hidden_layer_neurons

from config import first_activation_func, second_activation_func, third_activation_func, fourth_activation_func

class SimpleNN_4HiddenLayer(nn.Module):  
    def __init__(self):  
        super(SimpleNN_4HiddenLayer, self).__init__()
        
        input_size = input_dimension
        hidden1_size = first_hidden_layer_neurons
        hidden2_size = second_hidden_layer_neurons
        hidden3_size = third_hidden_layer_neurons
        hidden4_size =  fourth_hidden_layer_neurons
        output_size = 1
        
        self.fc1 = nn.Linear(input_size, hidden1_size)  # First layer  
        
        self.relu1 = first_activation_func 
        
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)  # Second layer 
        
        self.relu2 = second_activation_func
        
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)  # Third layer 
        
        self.relu3 = third_activation_func
        
        self.fc4 = nn.Linear(hidden3_size, hidden4_size)  # Fourth layer 
        
        self.relu4 = fourth_activation_func
        
        self.fc5 = nn.Linear(hidden4_size, output_size)  # Fifth layer 
        
    def forward(self, x):  
        x = self.fc1(x)  
        x = self.fc2(self.relu1.apply((x)))    
        x = self.fc3(self.relu2.apply((x))) 
        x = self.fc4(self.relu3.apply((x)))
        x = self.fc5(self.relu4.apply((x)))
 
        return x

class SimpleNN_1HiddenLayer(nn.Module):  
    def __init__(self):  
        super(SimpleNN_1HiddenLayer, self).__init__()
        
        input_size = input_dimension
        hidden1_size = first_hidden_layer_neurons
        output_size = 1
        
        self.fc1 = nn.Linear(input_size, hidden1_size)  # First layer   
        self.relu1 = first_activation_func 
        self.fc2 = nn.Linear(hidden1_size, output_size)  # Third layer 


    def forward(self, x):  
        x = self.fc2(self.relu1.apply((self.fc1(x))))
        return x 

class SimpleNN_2HiddenLayer(nn.Module):  
    def __init__(self):  
        super(SimpleNN_2HiddenLayer, self).__init__()
        
        self.fc1 = nn.Linear(input_dimension, first_hidden_layer_neurons)  # First layer   
        self.relu1 = first_activation_func 
        self.fc2 = nn.Linear(first_hidden_layer_neurons, second_hidden_layer_neurons)  # Second layer 
        self.relu2 = second_activation_func
        self.fc3 = nn.Linear(second_hidden_layer_neurons, 1)  # Third layer 


    def forward(self, x):  
        x = self.fc1(x)  
        x= self.fc2(self.relu1.apply((x)))    
        x = self.fc3(self.relu2.apply((x)))  

        return x