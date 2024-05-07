import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle') # use the custom style

# x_train is the input variable (size in 1000 square feet)
x_train = np.array([1.0, 1.3, 1.2, 1.6, 1.9, 1.5 ,2.0, 2.1, 2.4, 2.7, 2.6, 2.5, 2.5, 2.8])

# y train is the output variable (price in 1000 dollars)
y_train = np.array([300.0, 305.0, 302.0, 315.0, 330.0, 340.0, 360.0, 365.0, 370.0, 380.0, 385.0, 390.0, 395.0, 400.0])

print(f"x_train: {x_train}")
print(f"y_train: {y_train}")


# m is the number of training examples
print (f"x_train.shape: {x_train.shape}")
m = x_train.shape[0] # x_train.shape is (14,) so m = 14, x_train.shape[0] is the number of rows in x_train

for i in range (m):
    x_i = x_train[i] # input variable of the i-th training example
    y_i = y_train[i] # output variable of the i-th training example
    print(f"(x^{i}, y^{i}): ({x_i}, {y_i})") # print the i-th training example

# plot the data points
plt.scatter(x_train, y_train, marker = 'x', c="r")

plt.title("Housing Prices")
plt.ylabel("Price in 1000 Dollars")
plt.xlabel("Size in 1000 Square Feet")
plt.show()

w = 58.4578 # initial value of the weight
b = 235.2906 # initial value of the bias

# for x^0 => f_wb = w * x[0] +b
# for x^1 => f_wb = w * x[1] +b

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        w,b (scalar): model parameters
    Returns:
        f_wb (ndarray (m,)): model prediction
    """
    f_wb = np.zeros(m) # initialize the prediction
    for i in range(m):
        f_wb[i] = w* x[i] + b
    
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

# Plot the model prediction
plt.plot(x_train, tmp_f_wb, c="b", label = "Initial Model")

# Plot the data points
plt.scatter(x_train, y_train, marker = 'x', c="r", label = "Actual Values")

plt.title("Housing Prices")
plt.ylabel("Price in 1000 Dollars")
plt.xlabel("Size in 1000 Square Feet")
plt.legend()
plt.show()