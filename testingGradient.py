import numpy as np
import scipy as scp
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


def loss(x, x_0):
    return np.sum((x - x_0)**2)

def sample(a, n_eval):
    a = np.minimum(np.maximum(a, 0), 1)
    return bernoulli.rvs(a, size=(n_eval, 4))

def grad(alpha):
    right = np.mean(sample(alpha + epsilon, n_eval))
    left = np.mean(sample(alpha - epsilon, n_eval))
    grad = (loss(right, x_0) - loss(left, x_0)) / (2 * epsilon)
    grad = (value(alpha+epsilon) - value(alpha-epsilon)) / (2 * epsilon)
    return grad

def value(x):
    return loss(np.mean(sample(x, n_eval)), x_0)

x_0 = np.array([1., 0., 1., 1.])
epsilon = 0.05
n_eval = 10000
x = np.linspace(0, 1, 100)
y = np.vectorize(grad)(x)
y2 = np.vectorize(value)(x)

plt.figure()
plt.plot(x, y,'blue')
plt.plot(x, y2,'red')
plt.grid()
plt.show()