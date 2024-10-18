import numpy as np
import matplotlib.pyplot as plt

# Lorenz system
def xdot(x, y, z, a=10):
    return a*(y-x)

def ydot(x, y, z, b=28):
    return x*(b-z)-y

def zdot(x, y, z, c=8/3):
    return x*y-c*z

def lorenz(x, y, z, dt=0.01):
    x1 = x + xdot(x, y, z)*dt
    y1 = y + ydot(x, y, z)*dt
    z1 = z + zdot(x, y, z)*dt
    return x1, y1, z1

# def rk4(x, y, z, dt=0.01):
#     k1 = xdot(x, y, z)
#     l1 = ydot(x, y, z)
#     m1 = zdot(x, y, z)
#     k2 = xdot(x+k1*dt/2, y+l1*dt/2, z+m1*dt/2)
#     l2 = ydot(x+k1*dt/2, y+l1*dt/2, z+m1*dt/2)
#     m2 = zdot(x+k1*dt/2, y+l1*dt/2, z+m1*dt/2)
#     k3 = xdot(x+k2*dt/2, y+l2*dt/2, z+m2*dt/2)
#     l3 = ydot(x+k2*dt/2, y+l2*dt/2, z+m2*dt/2)
#     m3 = zdot(x+k2*dt/2, y+l2*dt/2, z+m2*dt/2)
#     k4 = xdot(x+k3*dt, y+l3*dt, z+m3*dt)
#     l4 = ydot(x+k3*dt, y+l3*dt, z+m3*dt)
#     m4 = zdot(x+k3*dt, y+l3*dt, z+m3*dt)
#     x1 = x + (k1+2*k2+2*k3+k4)*dt/6
#     y1 = y + (l1+2*l2+2*l3+l4)*dt/6
#     z1 = z + (m1+2*m2+2*m3+m4)*dt/6
#     return x

def lorenz_pattern(x, y, z, n=1000, dt=0.01):
    x_vals = [x]
    y_vals = [y]
    z_vals = [z]
    for i in range(n):
        x, y, z = lorenz(x, y, z, dt)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    return np.array([x_vals, y_vals, z_vals]).T

def plot_lorenz(x, y, z, n=1000):
    x_vals = [x]
    y_vals = [y]
    z_vals = [z]
    for i in range(n):
        x, y, z = lorenz(x, y, z)
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    return plot3d(x_vals, y_vals, z_vals)

def plot3d(x, y, z, fig=None):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    return ax

if __name__ == "__main__":
    plot_lorenz(1, 1, 1)