import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cap(val):
    val = 0.0 if val < 0.0 else val
    val = 1.0 if val > 1.0 else val
    return val


def pos(val):
    val = 0.0 if val < 0.0 else val
    return val


def plot_diagnostics():
    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(2, 5, 1)
    grid_vis = vis_act * w_vis_act
    grid_vis = grid_vis.reshape((dim, dim))
    ax.imshow(grid_vis, vmin=0.0, vmax=1.0)
    plt.imshow(grid_vis, vmin=0.0, vmax=1.0)
    plt.colorbar()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Spatial Frequency')
    ax.set_ylabel('Orientation')
    ax.set_title('Visual Input')

    ax = fig.add_subplot(2, 5, 2)
    grid_A = w_A.reshape((dim, dim))
    ax.imshow(grid_A, vmin=0.0, vmax=1.0)
    plt.imshow(grid_A, vmin=0.0, vmax=1.0)
    plt.colorbar()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Spatial Frequency')
    ax.set_ylabel('Orientation')
    ax.set_title('Connection Weights A')

    ax = fig.add_subplot(2, 5, 3)
    grid_B = w_B.reshape((dim, dim))
    ax.imshow(grid_B, vmin=0.0, vmax=1.0)
    plt.imshow(grid_B, vmin=0.0, vmax=1.0)
    plt.colorbar()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Spatial Frequency')
    ax.set_ylabel('Orientation')
    ax.set_title('Connection Weights B')

    ax = fig.add_subplot(2, 5, 4)
    grid_C = w_C.reshape((dim, dim))
    ax.imshow(grid_C, vmin=0.0, vmax=1.0)
    plt.imshow(grid_C, vmin=0.0, vmax=1.0)
    plt.colorbar()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Spatial Frequency')
    ax.set_ylabel('Orientation')
    ax.set_title('Connection Weights C')

    ax = fig.add_subplot(2, 5, 5)
    grid_D = w_D.reshape((dim, dim))
    ax.imshow(grid_D, vmin=0.0, vmax=1.0)
    plt.imshow(grid_D, vmin=0.0, vmax=1.0)
    plt.colorbar()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_xlabel('Spatial Frequency')
    ax.set_ylabel('Orientation')
    ax.set_title('Connection Weights D')

    plt.show()


def gen_stim(mean, cov, cat_lab, num_stim):

    x = []
    y = []
    cat = []

    for i in range(len(mean)):
        m = mean[i]
        c = cov[i]

        xy = np.random.multivariate_normal(m, c, num_stim[i])
        lab = [cat_lab[i]] * num_stim[i]

        x = np.append(x, xy[:, 0])
        y = np.append(y, xy[:, 1])
        cat = np.append(cat, lab)

    return {'cat': cat, 'x': x, 'y': y}


# Define Crossley et al. (2013) categories
mean_x = [72, 100, 100, 128]
mean_y = [100, 128, 72, 100]
cov = [[100, 0], [0, 100]]

mean_x = [x / 2.0 for x in mean_x]
mean_y = [x / 2.0 for x in mean_y]
cov = [[x[0] / 2.0, x[1] / 2.0] for x in cov]

n = 5
mean = [(mean_x[i], mean_y[i]) for i in range(len(mean_x))]

stimuli_A = gen_stim(mean, [cov, cov, cov, cov], [1, 2, 3, 4], [n, n, n, n])
stimuli_B = gen_stim(mean, [cov, cov, cov, cov], [2, 3, 4, 1], [n, n, n, n])
stimuli_A2 = gen_stim(mean, [cov, cov, cov, cov], [1, 2, 3, 4], [n, n, n, n])

stimuli = {
    'cat': np.append(stimuli_A['cat'], [stimuli_B['cat'], stimuli_A2['cat']]),
    'x': np.append(stimuli_A['x'], [stimuli_B['x'], stimuli_A2['x']]),
    'y': np.append(stimuli_A['y'], [stimuli_B['y'], stimuli_A2['y']])
}

# plot stimuli
x = stimuli_A['x']
y = stimuli_A['y']
cat = stimuli_A['cat']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x[cat == 1], y[cat == 1], '.r')
ax.plot(x[cat == 2], y[cat == 2], '.b')
ax.plot(x[cat == 3], y[cat == 3], '.g')
ax.plot(x[cat == 4], y[cat == 4], '.k')
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
plt.show()

# Simulate model
num_simulations = 1
num_stim = n * 4

vis_width = 25.0
alpha_critic = 0.1
alpha_actor = 0.1
stim = stimuli

data_record = {
    'cat': [],
    'x': [],
    'y': [],
    'resp': [],
    'accuracy': [],
    'delta': [],
    'w_sr': []
}

# static params:
dim = 100
w_base = 0.5
w_noise = 0.25
vis_amp = 1.0
w_vis_act = 0.15

# init buffers
vis_act = np.zeros(dim**2)
cumulative_weight_change = np.zeros(dim**2)
mean_weight_change = 0.0
w_A = np.random.normal(w_base, w_noise, dim**2)
w_B = np.random.normal(w_base, w_noise, dim**2)
w_C = np.random.normal(w_base, w_noise, dim**2)
w_D = np.random.normal(w_base, w_noise, dim**2)

act_A = 0
act_B = 0
act_C = 0
act_D = 0
discrim = 0
resp = 0

# init
v = 0.0
num_trials = len(stim['cat'])
for simulation in range(num_simulations):
    for trial in range(num_trials):

        print(simulation)
        print(trial)
        print("\n")

        cat = stim['cat'][trial]
        x = stim['x'][trial]
        y = stim['y'][trial]

        # compute input activation via radial basis functions
        vis_dist_x = 0.0
        vis_dist_y = 0.0
        for i in range(0, dim):
            for j in range(0, dim):
                vis_dist_x = x - i
                vis_dist_y = y - j

                vis_act[j + i * dim] = vis_amp * np.exp(
                    -(vis_dist_x**2 + vis_dist_y**2) / vis_width)

        # Compute unit activation via dot product
        act_A = np.inner(vis_act, w_A)
        act_B = np.inner(vis_act, w_B)
        act_C = np.inner(vis_act, w_C)
        act_D = np.inner(vis_act, w_D)

        # add noise to output units
        act_A += np.random.normal(w_base, w_noise, 1)
        act_B += np.random.normal(w_base, w_noise, 1)
        act_C += np.random.normal(w_base, w_noise, 1)
        act_D += np.random.normal(w_base, w_noise, 1)

        # Compute resp via max
        act_array = np.array([act_A, act_B, act_C, act_D])
        act_sort_ind = np.argsort(act_array, 0)

        # resp is ind + 1 because python indices are zero based
        resp = act_sort_ind[3][0] + 1

        # discrim is difference between the 2 most active units
        discrim = (act_array[3] - act_array[2]) / act_array[3]

        # Implement strong lateral inhibition
        act_A = act_A if resp == 1 else 0.0
        act_B = act_B if resp == 2 else 0.0
        act_C = act_C if resp == 3 else 0.0
        act_D = act_D if resp == 4 else 0.0

        # compute outcome
        r = 1.0 if cat == resp else -1.0

        # give random feedback between trials 300 and 600 as in Crossley et al.
        # (2013)
        if trial > 300 and trial <= 600:
            r = 1.0 if np.random.uniform(0.0, 1.0) > 0.5 else -1.0

        # Compute resp via max
        resp_obs = resp

        # compute outcome
        r_obs = r

        # compute prediction error
        delta = (r_obs - v)

        # update critic
        v += alpha_critic * delta

        # update actor --- stage 1
        for i in range(0, dim**2):
            if delta < 0:
                weight_change_A = alpha_actor * vis_act[
                    i] * act_A * delta * w_A[i]
                w_A[i] += weight_change_A

                weight_change_B = alpha_actor * vis_act[
                    i] * act_B * delta * w_B[i]
                w_B[i] += weight_change_B

                weight_change_C = alpha_actor * vis_act[
                    i] * act_C * delta * w_C[i]
                w_C[i] += weight_change_C

                weight_change_D = alpha_actor * vis_act[
                    i] * act_D * delta * w_D[i]
                w_D[i] += weight_change_D

            else:
                weight_change_A = alpha_actor * vis_act[i] * act_A * delta * (
                    1 - w_A[i])
                w_A[i] += weight_change_A

                weight_change_B = alpha_actor * vis_act[i] * act_B * delta * (
                    1 - w_B[i])
                w_B[i] += weight_change_B

                weight_change_C = alpha_actor * vis_act[i] * act_C * delta * (
                    1 - w_C[i])
                w_C[i] += weight_change_C

                weight_change_D = alpha_actor * vis_act[i] * act_D * delta * (
                    1 - w_D[i])
                w_D[i] += weight_change_D

            cumulative_weight_change[i] += (weight_change_A + weight_change_B +
                                            weight_change_C + weight_change_D)

            w_A[i] = cap(w_A[i])
            w_B[i] = cap(w_B[i])
            w_C[i] = cap(w_C[i])
            w_D[i] = cap(w_D[i])

        mean_weight_change = np.mean(cumulative_weight_change)

        # record keeping
        data_record['cat'].append(cat)
        data_record['x'].append(x)
        data_record['y'].append(0.0)
        data_record['resp'].append(resp_obs)
        data_record['accuracy'].append(cat == resp_obs)
        data_record['delta'].append(delta)
        data_record['w_sr'].append(mean_weight_change)

        trial += 1

# compute average over simulations
data_record_mean = {
    'accuracy': [0] * num_trials,
    'delta': [0] * num_trials,
    'w_sr': [0] * num_trials
}

for t in range(num_trials):
    for s in range(num_simulations):
        for key in data_record_mean.keys():
            data_record_mean[key][t] = data_record_mean[key][t] + data_record[
                key][t + s * num_trials]

for key in data_record_mean.keys():
    data_record_mean[key] = [
        x / float(num_simulations) for x in data_record_mean[key]
    ]

plot_diagnostics()
