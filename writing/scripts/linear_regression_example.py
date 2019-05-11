import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2019)

x = np.linspace(0, 2, 50)
y = 5 * x + 1 + np.random.normal(scale=1, size=x.shape)
fit = np.poly1d(np.polyfit(x, y, deg=1))


plt.plot(x, y, 'o', alpha=0.3, label='Data points')
plt.plot(x, fit(x), '--', label='Model')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')

import matplotlib2tikz

matplotlib2tikz.save(__file__ + ".tex")
plt.show()
