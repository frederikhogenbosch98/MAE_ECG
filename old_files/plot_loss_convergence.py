import matplotlib.pyplot as plt
import numpy as np

epoch = np.arange(1, 11)
loss_un = [0.1838104,0.1226304, 0.0860337, 0.0631535, 0.048219, 0.038120, 0.0313757, 0.0263643, 0.0225621, 0.0197242]
loss_cpd = [0.0840480, 0.0307184, 0.0160361, 0.0107587, 0.0081545, 0.0067279, 0.0058947, 0.0053201, 0.004941, 0.0046218]

plt.plot(epoch, loss_un, label='uncompressed')
plt.plot(epoch, loss_cpd, label='cpd(R=20)')
plt.title('validation loss for 10 epochs')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()