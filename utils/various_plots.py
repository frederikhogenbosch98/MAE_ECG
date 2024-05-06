import matplotlib.pyplot as plt
import numpy as np

def plot_compression_cp():
    R = [5, 10, 15, 20, 25]
    num_params = [6290, 11235, 16180, 21125, 26070]
    full_params = 2881921
    a = [full_params/i for i in num_params]
    plt.plot(R, a)
    plt.title('parameter compression ratio per rank (cp)')
    plt.xlabel('ranks')
    plt.ylabel('compression ratios')
    plt.xticks(R)
    plt.show()

def plot_accs_cp():
    R = [5, 10, 15, 20, 25]
    num_params = [6290, 11235, 16180, 21125, 26070]
    full_params = 2881921
    accuracies = [0.8209, 0.8406, 0.8482, 0.8322, 0.8443]
    
    # Calculate compression ratios
    a = [full_params / i for i in num_params]

    
    plt.plot(R, accuracies, 'o-', label='Accuracies')  # 'r-' is for red solid line
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    plt.tick_params('y')
    plt.ylim(0.80, 1.0)  # Set the range of the accuracy axis
    
    # Add horizontal lines for literature and 250 epochs benchmarks
    plt.axhline(0.9439, color='gray', linestyle='-', label='Literature (0.9439)')
    plt.axhline(0.9652, color='red', linestyle='-', label='250 Epochs (0.9652)')
    plt.title('CP accuracies with 100 MAE epoch and 25 CLASSIFIER epochs')
    plt.xticks(R)
    plt.legend() 
    plt.show()


def plot_compression_tucker():
    R = [5, 10, 15, 20, 25]
    num_params = [8626, 20621, 37566, 54845, 74300]
    full_params = 2881921
    a = [full_params/i for i in num_params]
    plt.plot(R, a)
    plt.title('parameter compression ratio per rank (tucker)')
    plt.xlabel('ranks')
    plt.ylabel('compression ratios')
    plt.xticks(R)
    plt.show()

def plot_accs_tucker():
    R = [5, 10, 15, 20, 25]
    num_params = [6290, 11235, 16180, 21125, 26070]
    full_params = 2881921
    accuracies = [0.8239, 0.8591, 0.8306, 0.8354, 0.8243]
    
    # Calculate compression ratios
    a = [full_params / i for i in num_params]

    
    plt.plot(R, accuracies, 'o-', label='Accuracies')  # 'r-' is for red solid line
    plt.xlabel('rank')
    plt.ylabel('Accuracies')
    plt.tick_params('y')
    plt.ylim(0.80, 1.0)  # Set the range of the accuracy axis
    
    # Add horizontal lines for literature and 250 epochs benchmarks
    plt.axhline(0.9439, color='gray', linestyle='-', label='Literature (0.9439)')
    plt.axhline(0.9652, color='red', linestyle='-', label='250 Epochs (0.9652)')
    plt.title('Tucker accuracies with 100 MAE epoch and 25 CLASSIFIER epochs')
    plt.xticks(R)
    plt.legend() 
    plt.show()

if __name__ == '__main__':
    plot_compression_cp()
    plot_accs_cp()
    plot_compression_tucker()
    plot_accs_tucker()