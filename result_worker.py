import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


def write_details(path, *args):
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(args)
    file.close()


def read_all_results() -> dict[str, pd.DataFrame]:
    # Directory where you want to search for CSV files
    directory = 'Results\\'

    # List to store CSV file paths
    csv_paths = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_paths.append(os.path.join(root, file))

    # Print all CSV file paths
    results = {}
    for csv_path in csv_paths:
        csv_config = csv_path.split("\\")[1]
        df = pd.read_csv(csv_path)
        results[csv_config] = df

    return results


def image_size_comparison(results: dict[str, pd.DataFrame]):
    Colors = ['b', 'g', 'r', 'purple']  # Color list should remain unchanged

    # Sort the keys of the results dictionary
    # Sorting the keys as integers (or by another criteria if needed)
    sorted_keys = sorted(results.keys(), key=int)

    # Iterate over the sorted keys
    for idx, image_size in enumerate(sorted_keys):
        # Get the data for the current image size
        result_data = results[image_size][:40]
        epochs = result_data['Epoch']
        # Make sure there's no leading space in 'Test Acc'
        test_acc = result_data[' Test Acc']

        # Plot Test Accuracy vs Epoch with a thinner line
        plt.plot(epochs, test_acc, color=Colors[idx % len(
            Colors)], label=f'{image_size}', linewidth=1)

        # Mark the highest accuracy with a red marker
        # Get the row with the max accuracy
        max_acc_epoch = result_data.loc[result_data[' Test Acc'].idxmax()]
        plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Test Acc'], color='black', marker='p',
                    s=100, label='Highest Acc' if Colors[idx % len(Colors)] == 'purple' else '')

    # Adding title and labels
    plt.title(
        "Image Size Comparison with config = 2Layer-100-100_lr-0.001_batchsize-512")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def neurons_comparison(results: dict[str, pd.DataFrame]):
    Colors = ['b', 'g', 'r']  # Color list should remain unchanged

    # Sort the keys of the results dictionary
    # Sorting the keys as integers (or by another criteria if needed)
    sorted_keys = sorted(results.keys(), key=str)

    # Iterate over the sorted keys
    for idx, neurons in enumerate(sorted_keys):
        # Get the data for the current neurons size
        result_data = results[neurons][:50]
        epochs = result_data['Epoch']
        # Make sure there's no leading space in 'Test Acc'
        test_acc = result_data[' Test Acc']

        # Plot Test Accuracy vs Epoch with a thinner line
        plt.plot(epochs, test_acc, color=Colors[idx % len(
            Colors)], label=f'{neurons}', linewidth=1)

        # Mark the highest accuracy with a red marker
        # Get the row with the max accuracy
        max_acc_epoch = result_data.loc[result_data[' Test Acc'].idxmax()]
        plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Test Acc'], color='black',
                    marker='p', s=100, label='Highest Acc' if Colors[idx % len(Colors)] == 'r' else '')

    # Adding title and labels
    plt.title(
        "Number of Neurons Comparsion with config = 2Layer_lr-0.001_batchsize-128_pixels-32")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def batchsize_comparison(results: dict[str, pd.DataFrame]):
    Colors = ['b', 'g',]  # Color list should remain unchanged

    # Sort the keys of the results dictionary
    # Sorting the keys as integers (or by another criteria if needed)
    sorted_keys = sorted(results.keys(), key=int)

    # Iterate over the sorted keys
    for idx, batchsize in enumerate(sorted_keys):
        # Get the data for the current neurons size
        result_data = results[batchsize][:100]
        epochs = result_data['Epoch']
        # Make sure there's no leading space in 'Test Acc'
        test_acc = result_data[' Test Acc']

        # Plot Test Accuracy vs Epoch with a thinner line
        plt.plot(epochs, test_acc, color=Colors[idx % len(
            Colors)], label=f'{batchsize}', linewidth=1)

        # Mark the highest accuracy with a red marker
        # Get the row with the max accuracy
        max_acc_epoch = result_data.loc[result_data[' Test Acc'].idxmax()]
        plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Test Acc'], color='black',
                    marker='p', s=100, label='Highest Acc' if Colors[idx % len(Colors)] == 'g' else '')

    # Adding title and labels
    plt.title("Batchsize Comparsion with config = 2Layer-100-100_lr-0.001_pixels-32")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def lr_comparison(results: dict[str, pd.DataFrame]):
    Colors = ['b', 'g',]  # Color list should remain unchanged

    # Sort the keys of the results dictionary
    # Sorting the keys as integers (or by another criteria if needed)
    sorted_keys = sorted(results.keys(), key=float)

    # Iterate over the sorted keys
    for idx, lr in enumerate(sorted_keys):
        # Get the data for the current neurons size
        result_data = results[lr][:40]
        epochs = result_data['Epoch']
        # Make sure there's no leading space in 'Test Acc'
        test_acc = result_data[' Test Acc']

        # Plot Test Accuracy vs Epoch with a thinner line
        plt.plot(epochs, test_acc, color=Colors[idx % len(
            Colors)], label=f'{lr}', linewidth=1)

        # Mark the highest accuracy with a red marker
        # Get the row with the max accuracy
        max_acc_epoch = result_data.loc[result_data[' Test Acc'].idxmax()]
        plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Test Acc'], color='black',
                    marker='p', s=100, label='Highest Acc' if Colors[idx % len(Colors)] == 'g' else '')

    # Adding title and labels
    plt.title(
        "Learning Rate Comparison with config = 2Layer-100-100_batchsize-512_pixels-128")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def layers_comparison(results: dict[str, pd.DataFrame]):
    # Color list should remain unchanged
    Colors = ['b', 'g', 'r', 'purple', 'orange', 'magenta']

    # Sort the keys of the results dictionary
    # Sorting the keys as integers (or by another criteria if needed)
    sorted_keys = sorted(results.keys(), key=str)

    # Iterate over the sorted keys
    for idx, layer in enumerate(sorted_keys):
        # Get the data for the current neurons size
        result_data = results[layer][:50]
        epochs = result_data['Epoch']
        # Make sure there's no leading space in 'Test Acc'
        test_acc = result_data[' Test Acc']

        # Plot Test Accuracy vs Epoch with a thinner line
        plt.plot(epochs, test_acc, color=Colors[idx % len(
            Colors)], label=f'{layer}', linewidth=1)

        # Mark the highest accuracy with a red marker
        # Get the row with the max accuracy
        max_acc_epoch = result_data.loc[result_data[' Test Acc'].idxmax()]
        plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Test Acc'], color='black',
                    marker='p', s=100, label='Highest Acc' if Colors[idx % len(Colors)] == 'g' else '')

    # Adding title and labels
    plt.title(
        "Number of Layers Comparison with config = lr-0.001_batchsize-128_pixels-32")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_best_config(results: dict[str, pd.DataFrame]):

    epochs = results['Epoch']
    test_acc = results[' Test Acc']
    train_acc = results[' Train Acc']
    loss = results[' BCE']

    # Plot Test Accuracy vs Epoch with a thinner line
    plt.plot(epochs, test_acc, color='r', linewidth=1, label='Test Acc/Epoch')
    plt.plot(epochs, train_acc, color='b', linewidth=1, label='Train Acc/Epoch')

    # Mark the highest accuracy with a red marker
    # Get the row with the max accuracy
    max_acc_epoch = results.loc[results[' Test Acc'].idxmax()]
    plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Test Acc'], color='black',
                marker='p', s=100, label='Highest Acc')
    
    max_acc_epoch = results.loc[results[' Train Acc'].idxmax()]
    plt.scatter(max_acc_epoch['Epoch'], max_acc_epoch[' Train Acc'], color='black',
                marker='p', s=100)

    # Adding title and labels
    plt.title(
        "Config = 2Layer-100-100_lr-0.001_batchsize-128_pixels-32")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()
    
    
    plt.plot(epochs, loss, color='g', linewidth=1, label='BCELoss/Epoch')
    
    min_loss_epoch = results.loc[results[' BCE'].idxmin()]
    plt.scatter(min_loss_epoch['Epoch'], min_loss_epoch[' BCE'], color='black',
                marker='p', s=100, label=f'Minimum Loss = {min_loss_epoch[" BCE"]:.2f}')

    # Adding title and labels
    plt.title(
        "Config = 2Layer-100-100_lr-0.001_batchsize-128_pixels-32")
    plt.xlabel("Epoch")
    plt.ylabel("BCELoss")

    # Display the grid
    plt.grid(True)

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    results = read_all_results()

    image_size_results = {}
    for config, result in results.items():
        if '2Layer-100-100_lr-0.001_batchsize-512' in config:
            image_size = int(config[config.find('pixels')+len('pixels-'):])
            image_size_results[image_size] = result
    # image_size_comparison(image_size_results)

    neurons_2_layer_results = {}
    for config, result in results.items():
        if '2Layer' in config and 'pixels-32' in config and 'batchsize-128' in config:
            neurons = config[len('2Layer-'):config.find('_')]
            neurons_2_layer_results[neurons] = result
    # neurons_comparison(neurons_2_layer_results)

    batch_size_results = {}
    for config, result in results.items():
        if '2Layer-100-100_lr-0.001' in config and 'pixels-32' in config:
            batchsize = int(
                config[config.find('batchsize')+len('batchsize-'):config.find('_pixels')])
            batch_size_results[batchsize] = result
    # batchsize_comparison(batch_size_results)

    lr_results = {}
    for config, result in results.items():
        if '2Layer-100-100' in config and 'batchsize-512_pixels-128' in config:
            lr = float(config[config.find('lr')+len('lr-')                       :config.find('_batchsize')])
            lr_results[lr] = result
    # lr_comparison(lr_results)

    layers_results = {}
    for config, result in results.items():
        if 'lr-0.001_batchsize-128_pixels-32' in config:
            layer = config[:config.find('_lr')]
            layers_results[layer] = result
    # layers_comparison(layers_results)

    plot_best_config(
        results['2Layer-100-100_lr-0.001_batchsize-128_pixels-32'])
