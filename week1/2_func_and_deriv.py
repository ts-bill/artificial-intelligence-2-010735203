from typing import Callable
import numpy as np

def square(x:np.ndarray) -> np.ndarray:
    return np.power(x,2)

# def sq2(x):
#     return np.power(x,2)

def leaky_relu(x:np.ndarray):
    return np.maximum(0.2*x,1*x)

def relu(x:np.ndarray):
    return np.maximum(0,x)

def deriv(func:Callable[[np.ndarray],np.ndarray],input_:np.ndarray,delta:float=0.001):
    return (func(input_+delta)-func(input_-delta))/(2*delta)

# func= square >> x^2 , deriv=2x
# ex. if x=3 deriv=2(3)=6

deriv_x=deriv(square,10)
print(deriv_x)