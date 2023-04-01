import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as integrate
from scipy.misc import derivative

# Calculating derivatives and integrals using scipy iterative methods
f = lambda x: x ** 3 + 1/x
dx = 1e-6

differential = derivative(f, x0=1, dx=dx)
integral = integrate.quad(f, a=1, b=3)[0]

print("Derivative at the point x0 = 1: ", differential)
print("Definite integral from 1 to 3: ", integral)


# Calculating the derivative of a function using the Finite Differences method
h = 0.05
x = np.arange(-3, 3, h)
y = np.tanh(x)

range_x = x[:-1]

finite_difference = np.diff(y)/h
direct_derivative = 1 - np.square(np.tanh(range_x))

plt.plot(range_x, finite_difference, '--', label='Approximation')
plt.plot(range_x, direct_derivative, label='Real Curve')
plt.legend()
plt.show()

MAE = sum(abs(direct_derivative - finite_difference)) / len(range_x)
print("Mean Absolute Error for the Finite Difference method: ", MAE)


# Solving a ODE in the range -1, 2 using the Euler method
h = 1e-3
x = np.arange(0, 1, h)

s = np.zeros(len(x))
s[0] = 1.2

f = lambda X, S: (2 * S) - X
analytical = lambda X: (2 * X + 1) / 4 + np.power(np.e, 2 * X)

for i in range(0, len(x) - 1):
    s[i + 1] = s[i] + h * f(x[i], s[i])

plt.plot(x, s, '--', label='Finite Solution')
plt.plot(x, analytical(x), 'g', label='Analytical Solution')
plt.xlabel('x')
plt.ylabel('f(x) = y')
plt.legend()
plt.grid()
plt.show()
