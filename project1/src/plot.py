#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Load data.
data = np.loadtxt('../output/vmc.txt', skiprows=1)
alpha, beta, E, E2 = [data[:, i] for i in range(4)]

# Compute minimum energy.
min_i = np.argmin(E)
print('E(alpha={}) = {}'.format(alpha[min_i], E[min_i]))

# Compute minimum variance.
var = E2 - E**2
min_i_var = np.argmin(var)
print('Var(alpha={}) = {}'.format(alpha[min_i_var], var[min_i_var]))



# Plot
fig, ax = plt.subplots(nrows=2, sharex=True)

ax[0].plot(alpha, E)
ax[0].plot([alpha[min_i]], [E[min_i]], 'ro')
ax[0].set_ylim([-0.5, 10])
ax[0].set_ylabel(r'$\langle H\rangle$', fontsize=20)
ax[0].set_title(r'Ground state energy as function of variational parameter $\alpha$')

ax[1].plot(alpha, var)
ax[1].plot([alpha[min_i_var]], [var[min_i_var]], 'ro')
ax[1].set_ylim([-0.5, 10])
ax[1].set_xlabel(r'$\alpha$', fontsize=20)
ax[1].set_ylabel(r'$\sigma^2_{E}$', fontsize=20)
ax[1].set_title(r'Variance as function of variational parameter $\alpha$')

plt.show()
