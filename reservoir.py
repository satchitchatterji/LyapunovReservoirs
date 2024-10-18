import numpy as np

class Reservoir:
    def __init__(self, N, in_shape, out_shape, sparsity=0.9, spectral_radius=1.5):
        self.N = N
        self.sparsity = sparsity
        self.W = np.random.rand(N,N)
        self.W[self.W > sparsity] = 0
        # scale spectral radius to 1
        self.W = (self.W / (np.max(np.abs(np.linalg.eigvals(self.W)))))*spectral_radius

        self.x = np.zeros(N)
        self.W_in = np.random.rand(N, in_shape)
        self.W_out = np.random.rand(out_shape, N)
        self.b = np.random.rand(N)

        self.collect = []

    def update(self, in_pattern):
        self.x = np.tanh(self.W.dot(self.x) + np.dot(self.W_in, in_pattern) + self.b)

    def train_out(self, in_pattern, out_pattern, washout):
        for i in range(len(in_pattern)):
            self.update(in_pattern[i])
            if i > washout:
                self.collect.append(self.x)

        self.collect = np.array(self.collect)
        self.W_out = np.linalg.pinv(self.collect).dot(out_pattern[washout+1:]).T
        # print("W_out shape: ", self.W_out.shape)

    def predict(self, in_pattern_point):
        self.update(in_pattern_point)
        return self.W_out.dot(self.x)

    def reset(self):
        self.x = np.zeros(self.N)
        self.collect = []