import matplotlib.pyplot as plt
import torch
import numpy as np

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    return x * cdf


x = torch.linspace(-3, 3, steps=60)
plt.figure(figsize=(10,6))
plt.plot(x, gelu(x))
plt.title('GELU activation function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()