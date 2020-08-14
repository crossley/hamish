import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_steps = 900
amp = 0.89
I = np.concatenate(
    (np.zeros(n_steps // 3), amp * np.ones(n_steps // 3), np.zeros(n_steps // 3)))

## Integrate and fire neuron
b = 0.1
V = b * np.ones(n_steps)

for t in range(n_steps - 1):
    V[t + 1] = V[t] + (b - V[t]) + I[t] 
    if V[t] >= 1:
        V[t + 1] = 0

plt.plot(V)
plt.plot(I)
plt.show()
