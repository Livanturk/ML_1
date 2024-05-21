import numpy as np
import matplotlib.pyplot as plt
from plt_logistic_loss import plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
plt.style.use('./deeplearning.mplstyle')
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error


soup_bowl()

# Training Data
x_train = np.array([0., 1, 2, 3, 4, 5], dtype= np.longdouble)
y_train = np.array([0, 0, 0, 1, 1, 1], dtype = np.longdouble)
plt_simple_example(x_train, y_train)
plt.show()

plt.close("all")
plt_logistic_squared_error(x_train, y_train)
plt.show()