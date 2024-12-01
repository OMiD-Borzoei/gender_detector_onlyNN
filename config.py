from torch import nn 
from actiavtion_functions import CustomReLU

lr = 0.001
batch_size = 128  # 64
num_epochs = 50

input_width = 32
input_height = 32

input_dimension = input_width * input_height * 3

test_portion = 0.25

first_hidden_layer_neurons = input_dimension//8
second_hidden_layer_neurons = first_hidden_layer_neurons//4
third_hidden_layer_neurons = first_hidden_layer_neurons//4
fourth_hidden_layer_neurons = third_hidden_layer_neurons//4

first_activation_func = CustomReLU
second_activation_func = CustomReLU
third_activation_func = CustomReLU
fourth_activation_func = CustomReLU

detail_view = True

# model_detail = f"4Layer-{first_hidden_layer_neurons}-{second_hidden_layer_neurons}-{third_hidden_layer_neurons}-{fourth_hidden_layer_neurons}"

#model_detail = f"1Layer-{first_hidden_layer_neurons}"

model_detail = f"2Layer-{first_hidden_layer_neurons}-{second_hidden_layer_neurons}"

result_path = f'Results/{model_detail}_lr-{lr}_batchsize-{batch_size}_pixels-{input_width}'
result_name = 'detail.csv'

result_full_path = result_path + '/' + result_name
