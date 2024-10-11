import numpy as np
from matplotlib import pyplot as plt


# Define the four basis functions
def phi_1(x_i):
    return 1 - 3 * x_i ** 2 + 2 * x_i ** 3


def phi_2(x_i):
    return x_i * (x_i - 1) ** 2


def phi_3(x_i):
    return 3 * x_i ** 2 - 2 * x_i ** 3


def phi_4(x_i):
    return (x_i ** 2) * (x_i - 1)


# Define the piecewise functions based on basis functions
def phi_2i_minus_1(x, x_i, h):
    return np.piecewise(
        x,
        [x < x_i - h, (x >= x_i - h) & (x <= x_i), (x > x_i) & (x <= x_i + h), x > x_i + h],
        [0, lambda x: phi_3((x - x_i + h) / h), lambda x: phi_1((x - x_i) / h), 0]
    )


def phi_2i(x, x_i, h):
    return np.piecewise(
        x,
        [x < x_i - h, (x >= x_i - h) & (x <= x_i), (x > x_i) & (x <= x_i + h), x > x_i + h],
        [0, lambda x: h * phi_4((x - x_i + h) / h), lambda x: h * phi_2((x - x_i) / h), 0]
    )


# Function to approximate the piecewise polynomial
def approximate(n, x_full, u):
    h = x_full / (n - 1)
    xi = np.linspace(0, x_full, n)  # Equally spaced nodes
    x = np.linspace(0, x_full, 1000)

    # Initialize the polynomial approximation w
    w = np.zeros_like(x)

    # Add contributions from each basis function
    for i in range(n):
        w += u[2 * i] * phi_2i_minus_1(x, xi[i], h) + u[2 * i + 1] * phi_2i(x, xi[i], h)

    return x, w


# Parameters
n = 5
x_full = 2 * np.pi

# Define u with sin(x) values and derivatives cos(x) at the nodes
u = np.zeros(2 * n)
u[::2] = np.sin(np.linspace(0, x_full, n))  # Values of sin at nodes
u[1::2] = np.cos(np.linspace(0, x_full, n))  # Derivatives of sin at nodes

# Approximate and plot
x, w = approximate(n, x_full, u)

# Plot the result and compare with sin(x)
plt.plot(x, w, label='Piecewise Polynomial')
plt.plot(x, np.sin(x), 'r--', label='sin(x)', alpha=0.7)
plt.legend()
plt.xlabel('x')
plt.ylabel('w(x)')
plt.title('Piecewise Polynomial Approximation of sin(x)')
plt.grid(True)
plt.show()