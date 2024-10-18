from patterngen import lorenz_pattern, plot3d
from reservoir_pathak import Reservoir
import numpy as np
import matplotlib.pyplot as plt
import config

plot_3d_traj = True

# np.random.seed(config.seed)
pattern_full = lorenz_pattern(1, 1, 1, n=config.n_pre + config.n_train + config.n_test, dt=config.dt)
pattern_pre = pattern_full[:config.n_pre]
pattern_train = pattern_full[config.n_pre:config.n_pre + config.n_train]
pattern_test = pattern_full[config.n_pre + config.n_train:]

reservoir = Reservoir(N=config.N, in_shape=3, out_shape=3, 
                      sparsity=config.sparsity, spectral_radius=config.spectral_radius)

reservoir.initialize_hidden(pattern_pre)
reservoir.train_out(pattern_train[:-1], pattern_train[1:], washout=config.washout, beta=config.beta)

# reservoir.set_initial_out(pattern_test[0])



predicts = [reservoir.out]
for i in range(config.n_test):
    predicts.append(reservoir.udpate_auto())

predicts = np.array(predicts)

washout=config.washout

if plot_3d_traj:
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pattern_test[:,0], pattern_test[:,1], pattern_test[:,2], linestyle="--", label="Target")
    ax.plot(predicts[washout:,0], predicts[washout:,1], predicts[washout:,2], label="Prediction", alpha=0.5)
    # limit axes
    ax.set_xlim([-20,20])
    ax.set_ylim([-30,30])
    ax.set_zlim([0,50])

    plt.legend()
    plt.show()

# exit()
# plot all 3 axes separately

# xlim = [0,1/dt*25]
xlim = [0,25]
fig = plt.figure()
ax = fig.add_subplot(311)
x_vals = np.arange(0, config.n_test+1, 1)
x_vals = x_vals*config.dt
ax.plot(x_vals, pattern_test[:,0], label="Target")
ax.plot(x_vals[washout:], predicts[washout:,0], label="Prediction")
ax.set_ylim([-20,20])
ax.set_xlim(xlim)
plt.legend()

ax = fig.add_subplot(312)
ax.plot(x_vals, pattern_test[:,1], label="Target")
ax.plot(x_vals[washout:], predicts[washout:,1], label="Prediction")
ax.set_ylim([-30,30])
ax.set_xlim(xlim)

ax = fig.add_subplot(313)
ax.plot(x_vals, pattern_test[:,2], label="Target")
ax.plot(x_vals[washout:], predicts[washout:,2], label="Prediction")
ax.set_ylim([0,50])
ax.set_xlim(xlim)


plt.show()