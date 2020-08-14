import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tau = 1
C = 100.0
vr = -75.0
vt = -45.0
k = 1.2
a = 0.01
b = 5.0
c = -56.0
d = 130.0
vpeak = 60.0
E = 950.0

n_steps = 1000
tan_v = np.zeros(n_steps)
tan_u = np.zeros(n_steps)
tan_spikes = np.zeros(n_steps)
tan_output = np.zeros(n_steps)
spike_length = np.zeros(n_steps)

pf_tan_mod = 1.0
pause_mod_amp = 1.0
pause_mod = np.zeros(n_steps)
pf_tan_act = np.zeros(n_steps)

tan_v[0] = vr

for i in range(n_steps-1):
    noise = np.random.normal(0, 1)

    tan_v[i + 1] = tan_v[i] + tau * (k * (tan_v[i] - vr) * (tan_v[i] - vt) -
                                     tan_u[i] + E + pf_tan_act[i] + noise) / C

    tan_u[i +
          1] = tan_u[i] + tau * a * (b * (tan_v[i] - vr) - tan_u[i] +
                                     pf_tan_mod * pause_mod_amp * pause_mod[i])

    if tan_v[i + 1] >= vpeak:
        tan_v[i] = vpeak
        tan_v[i + 1] = c
        tan_u[i + 1] = tan_u[i + 1] + d
        tan_spikes[i + 1] = 1

    else:
        tan_spikes[i + 1] = 0


plt.plot(np.arange(0, n_steps, 1), tan_v)
plt.show()
