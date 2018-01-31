#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Load data.
data = np.loadtxt('../output/vmc.txt', skiprows=1)
alpha, beta, E, E2 = [data[:, i] for i in range(4)]

# Compute minimum variance.
var = E2 - E**2
min_i = np.argmin(var)
print('Var(alpha={}) = {}'.format(alpha[min_i], var[min_i]))


# Plot
plt.plot(alpha, var)
plt.plot([alpha[min_i]], [var[min_i]], 'ro')
plt.ylim([-0.5, 10])
plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel(r'$\sigma^2_{E}$', fontsize=20)
plt.title(r'Variance as function of variational parameter $\alpha$')
plt.show()
