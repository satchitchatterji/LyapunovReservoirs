from reservoir import Reservoir
import matplotlib.pyplot as plt
import numpy as np

# 2D pattern
reservoir = Reservoir(N=2, in_shape=2, out_shape=2)
n = 10000
start = 0
end = 20
patternx = np.sin(np.linspace(start,end,n))

# square wave with period 20
# patterny = np.array([0 if i % 200 < 100 else 1 for i in range(n)])
patterny = np.cos(np.linspace(start,end,n))

pattern = np.array([patternx,patterny]).T
# pattern2 = np.array([patterny,patternx]).T
# pattern = pattern + 0.1*np.random.randn(n,2)

reservoir.train_out(pattern, pattern, washout=1000)

neuron_outs = []

for i in range(n):  
    reservoir.update(pattern[i])
    neuron_outs.append(reservoir.x)

neuron_outs = np.array(neuron_outs)

predicts = []
for i in range(n):
    predicts.append(reservoir.predict(pattern[i]))

predicts = np.array(predicts)
plt.plot(predicts, linestyle="--", color="red", label="Prediction")

plt.plot(neuron_outs, alpha=0.5)
plt.plot(pattern, linestyle="--", color="black", label="Target")

plt.legend()

plt.show()

plt.plot(patternx, patterny, label="Target")
# plt.plot(patterny, patternx, label="Target")
plt.plot(predicts[:,0], predicts[:,1], label="Prediction")
plt.plot(neuron_outs[:,0], neuron_outs[:,1], label="Neurons")

plt.legend()
plt.show()