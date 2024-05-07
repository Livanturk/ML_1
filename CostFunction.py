import numpy as np
import matplotlib.widgets 
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
x_train = np.array([1.0, 1.3, 1.2, 1.6, 1.9, 1.5 ,2.0, 2.1, 2.4, 2.7, 2.6, 2.5, 2.5, 2.8])
# y train is the output variable (price in 1000 dollars)
y_train = np.array([300.0, 305.0, 302.0, 315.0, 330.0, 340.0, 360.0, 365.0, 370.0, 380.0, 385.0, 390.0, 395.0, 400.0])

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0] # number of training examples
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    
    total_cost = cost_sum / (2 * m)

    return total_cost

fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
plt.show()