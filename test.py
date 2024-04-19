import json
import matplotlib.pyplot as plt
from plots import load
import numpy as np


a = [100, 40, 50, 12, -45]

b = np.fft.rfft(a)

plt.plot(a)
plt.plot(b)
plt.show()
