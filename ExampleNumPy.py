import numpy as np
import time

#NumPy rouitnes which allocate memory and fill arrays with value
a = np.zeros(4) ; print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,)) ; print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4) ; print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# NumPy routines which allocate memory and fill arrays with value 
#but do not accept shape as input argument
a = np.arange(4.) ; print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4) ;  print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]) ; print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


def my_dot(a,b):
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x = 0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x

a = np.array([1,2,3,4])
b = np.array([-1,4,3,2])

print(f"my_dot(a,b) = {my_dot(a,b)}")

c = np.dot(a,b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")
