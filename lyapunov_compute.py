from reservoir_pathak import Reservoir
import numpy as np
import matplotlib.pyplot as plt
from patterngen import lorenz_pattern
import config
from tqdm import trange

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

n_runs = 10

code_1_lyapunov = []
code_2_lyapunov = []
reservoir = Reservoir(N=config.N, in_shape=3, out_shape=3, sparsity=config.sparsity, spectral_radius=config.spectral_radius)

for i in trange(n_runs):
    # np.random.seed(config.seed)
    reservoir.reset()
    lorenz_start_random = np.random.rand(3) * 2 - 1
    pattern_full = lorenz_pattern(lorenz_start_random[0], lorenz_start_random[1], lorenz_start_random[2], n=config.n_pre + config.n_train + config.n_test, dt=config.dt)
    pattern_pre = pattern_full[:config.n_pre]
    pattern_train = pattern_full[config.n_pre:config.n_pre + config.n_train]
    pattern_test = pattern_full[config.n_pre + config.n_train:]

    # reservoir = Reservoir(N=config.N, in_shape=3, out_shape=3, sparsity=config.sparsity, spectral_radius=config.spectral_radius)
    reservoir.initialize_hidden(pattern_pre)
    reservoir.train_out(pattern_train[:-1], pattern_train[1:], washout=config.washout, beta=config.beta)

    # Lyapunov exponent computation parameters
    num_lyaps = 3
    reservoir.num_lyaps = num_lyaps
    reservoir.initialize_delta()
    reservoir.set_initial_out(pattern_train[-1])

    ####################### CODE 1 #######################

    def compute_jacobian(r, W, W_in, W_out):
        """Compute the Jacobian of the reservoir update rule."""
        z = W.dot(r) + W_in.dot(W_out.dot(r))
        diag_term = 1 - np.tanh(z) ** 2  # Derivative of tanh is 1 - tanh^2
        J = np.diag(diag_term).dot(W + W_in.dot(W_out))  # Jacobian matrix
        return J

    def lyapunov_exponents(W, W_in, W_out, r0, delta_r0, T):
        """Compute Lyapunov exponents over T time steps."""
        N = len(r0)
        r = r0.copy()  # Initial reservoir state
        delta_r = delta_r0.copy()  # Initial perturbation
        # Initialize Lyapunov sums
        lyapunov_sums = np.zeros(N)

        for t in range(T):
            # Update the reservoir state
            # reservoir.update(pattern_test[t+1])
            reservoir.udpate_auto()
            r_next = reservoir.x
            
            # Compute Jacobian at the current time step
            J = compute_jacobian(r, W, W_in, W_out)
            
            # Evolve the perturbation using the Jacobian
            delta_r_next = J.dot(delta_r)

            # Perform QR decomposition for orthogonalization
            Q, R = np.linalg.qr(delta_r_next)

            # Update Lyapunov sum with the log of the diagonal elements of R
            lyapunov_sums += np.log(np.abs(np.diag(R)))

            # Update states
            r = r_next
            delta_r = Q  # Use the orthogonalized perturbations for the next step

        # Calculate Lyapunov exponents as the average growth rate
        lyapunov_exponents = lyapunov_sums / T
        return lyapunov_exponents

    # Parameters
    eps = 1e-8  # Small perturbation magnitude

    # Small perturbation to initial state
    delta_r0 = np.eye(config.N) * eps  # Small perturbation, identity matrix scaled by epsilon

    # Compute the Lyapunov exponents
    lyapunov_exp = lyapunov_exponents(reservoir.W, reservoir.W_in, reservoir.W_out, reservoir.x, delta_r0,
                                    config.n_test)
    # Output the Lyapunov exponents
    # plt.plot(lyapunov_exp[:3], label="Code 1 (time=1)", marker="o")

    code_1_lyapunov.append(lyapunov_exp[:3])

    ####################### CODE 2 #######################

    reservoir.set_initial_out(pattern_train[-1])
    # Test different norm_time values
    norm_times = [1]  # Different normalization frequencies to try
    results = {}

    for norm_time in norm_times:
        lyapunov_sums = np.zeros(num_lyaps)
        steps = 1000  # Number of steps to predict

        for i in range(steps):
            # Update reservoir state and evolve delta
            reservoir.udpate_auto()
            reservoir.update_delta()

            # Perform QR normalization every `norm_time` steps
            if i % norm_time == 0:
                R_ii = reservoir.qr_normalize_delta()
                lyapunov_sums += R_ii

        # Compute Lyapunov exponents
        lyapunov_exponents = lyapunov_sums / (steps / norm_time)
        results[norm_time] = lyapunov_exponents

    # # Plot the results for different norm_time values
    # for norm_time, lyapunov_exponents in results.items():
    #     plt.plot(lyapunov_exponents[:3], label=f'norm_time={norm_time}', marker="o")

    code_2_lyapunov.append(results[1][:3])


code_1_mean = np.mean(code_1_lyapunov, axis=0)
code_1_std = np.std(code_1_lyapunov, axis=0)
code_2_mean = np.mean(code_2_lyapunov, axis=0)
code_2_std = np.std(code_2_lyapunov, axis=0)

print("Actual:", np.array([0.91,0.00,-14.6]))
print("Code 1:", [str(np.round(x,4))+' ± '+str(np.round(y,4)) for x,y in zip(code_1_mean, code_1_std)])
print("Code 2:", [str(np.round(x,4))+' ± '+str(np.round(y,4)) for x,y in zip(code_2_mean, code_2_std)])

plt.title("Lyapunov Exponents for Different norm_time Values")
plt.plot([0.91,0.0,-14.6], label="Actual", marker="o")

plt.errorbar(np.arange(3), code_1_mean, yerr=code_1_std, label="Code 1", marker="o")
plt.errorbar(np.arange(3), code_2_mean, yerr=code_2_std, label="Code 2", marker="o")

# print(np.mean(np.sum(reservoir.W>0, axis=0)))

plt.xlabel("Exponent Index")
plt.ylabel("Lyapunov Exponent")
# plt.yscale("log")
plt.legend()
# plt.savefig("lyapunov.png", bbox_inches="tight")
plt.show()

#plot distribution of lyapunov exponents (3 parallel plots)
plt.figure()

plt.subplot(1, 3, 1)
plt.hist(np.array(code_1_lyapunov)[:,0], bins=10, alpha=0.5, label="Code 1")
plt.hist(np.array(code_2_lyapunov)[:,0], bins=10, alpha=0.5, label="Code 2")
plt.legend(["Code 1", "Code 2"])

plt.subplot(1, 3, 2)
plt.hist(np.array(code_1_lyapunov)[:,1], bins=10, alpha=0.5)
plt.hist(np.array(code_2_lyapunov)[:,1], bins=10, alpha=0.5)

plt.subplot(1, 3, 3)
plt.hist(np.array(code_1_lyapunov)[:,2], bins=10, alpha=0.5)
plt.hist(np.array(code_2_lyapunov)[:,2], bins=10, alpha=0.5)

# plt.savefig("lyapunov_dist.png", bbox_inches="tight")
plt.show()

# plot 3 lyapanov exponents over time, 1 row, 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15,5))

for i in range(3):
    axs[i].plot(np.array(code_1_lyapunov)[:,i], label="Code 1")
    axs[i].plot(np.array(code_2_lyapunov)[:,i], label="Code 2")
    axs[i].set_title(f"Lyapunov Exponent {i+1}")
    axs[i].set_xlabel("Run")
    axs[i].set_ylabel("Lyapunov Exponent")
    axs[i].legend()
    axs[i].grid()
plt.tight_layout()
plt.show()