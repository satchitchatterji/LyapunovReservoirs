import numpy as np
from scipy import sparse
from scipy.linalg import qr

class Reservoir:
    def __init__(self, N, in_shape, out_shape, sparsity=0.9, spectral_radius=1.5):
        self.N = N
        self.sparsity = sparsity
        self.in_shape = in_shape
        self.out_shape = out_shape
        # Convert sparse matrix to ndarray
        self.W = sparse.random(N, N, density=sparsity).toarray()  # Force to ndarray
        self.W = self.W - 0.5
        
        self.W = (self.W / self.get_spectral_radius()) * spectral_radius
        self.x = np.zeros(N)
        self.W_in = np.random.rand(N, in_shape)
        
        #### from uvadlc2
        weights_input = np.zeros((N, in_shape))
        q = int(N / in_shape)
        for i in range(0, in_shape):
            weights_input[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1
        self.W_in = weights_input
        
        self.W_out = np.random.rand(out_shape, N)
        self.collect = []
        self.out = np.zeros(out_shape)

        # For Lyapunov exponents
        self.delta = None  # Perturbation matrix
        self.num_lyaps = 10  # Number of Lyapunov exponents to compute

    def augment_hidden(self):
        self.x[::2] = np.pow(self.x[::2], 2.0)

    def get_spectral_radius(self):
        eigenvalues, _ = sparse.linalg.eigs(self.W)
        return np.max(np.abs(eigenvalues))

    def update(self, in_pattern):
        self.x = np.tanh(self.W.dot(self.x) + np.dot(self.W_in, in_pattern))
        self.augment_hidden()

    def tikhonov_solve(self, A, b, beta):
        return np.linalg.inv(A.T.dot(A) + beta * np.eye(A.shape[1])).dot(A.T).dot(b)

    def initialize_hidden(self, sequence):
        self.x = np.zeros(self.N)
        for s in sequence:
            self.x = np.tanh(self.W @ self.x + self.W_in @ s)
        return self.x

    def train_out(self, in_pattern, out_pattern, washout, beta=0):
        for i in range(len(in_pattern)-1):
            # ignoring washout for now, assuming self.initialized_hidden() was called
            self.update(in_pattern[i])
            self.collect.append(self.x)

        self.collect = np.array(self.collect)
        self.W_out = self.tikhonov_solve(self.collect, out_pattern[1:], beta).T
        self.out = self.W_out.dot(self.x)

    def set_initial_out(self, out):
        self.out = out

    def predict(self, in_pattern_point):
        self.update(in_pattern_point)
        return self.W_out.dot(self.x)

    def udpate_auto(self):
        self.update(self.out)
        self.out = self.W_out.dot(self.x)
        return self.out

    def reset(self):
        self.x = np.zeros(self.N)
        self.collect = []
        self.reset_out()

    def reset_out(self):
        self.out = np.zeros(self.out_shape)

    def initialize_delta(self):
        """Initialize random orthogonal perturbation vectors (delta)."""
        self.delta = np.linalg.qr(np.random.rand(self.N, self.num_lyaps))[0]

    def update_delta(self):
        """Evolve the perturbation vectors (delta) along with the system's dynamics."""
        K1 = self.W + self.W_in @ self.W_out  # Ensure K1 has shape (300, 300)
        K2 = 2 * (self.W_in @ self.W_out)     # Ensure K2 has shape (300, 300)

        # Convert all relevant arrays to ndarray (if not already)
        K1 = np.asarray(K1)
        K2 = np.asarray(K2)
        self.x = np.asarray(self.x)
        self.delta = np.asarray(self.delta)

        # First, compute K1 @ delta
        K1_delta = K1 @ self.delta  # Shape: (300, 10)

        # Next, compute the element-wise product between self.x and self.delta
        x_delta = self.x[:, np.newaxis] * self.delta  # Shape: (300, 10)

        # Now, compute K2 @ (self.x[:, np.newaxis] * self.delta)
        K2_x_delta = K2 @ x_delta  # Shape: (300, 10)

        # Finally, combine the results and apply the element-wise scaling by (1 - self.x ** 2)
        self.delta = (1 - self.x ** 2)[:, np.newaxis] * (K1_delta + K2_x_delta)

    def qr_normalize_delta(self):
        """QR decomposition to normalize the perturbation vectors and prevent numerical overflow."""
        self.delta, R = qr(self.delta, mode='economic')
        return np.log(np.abs(np.diag(R)))
