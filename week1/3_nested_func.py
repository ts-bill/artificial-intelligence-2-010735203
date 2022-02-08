import numpy as np

from typing import Callable, List

def square(x:np.ndarray) -> np.ndarray:
    return np.power(x,2)

def leaky_relu(x:np.ndarray):
    return np.maximum(0.2*x,x)

def relu(x:np.ndarray):
    return np.maximum(0,x)

def deriv(func:Callable[[np.ndarray],np.ndarray],input_:np.ndarray,delta:float=0.01):
    return (func(input_+delta)-func(input_-delta))/(2*delta)

Array_function=Callable[[np.ndarray],np.ndarray]

Chain = List[Array_function]

def chain_length_2(chain:Chain,x:np.ndarray)->np.ndarray:
    assert len(chain) == 2 #length of input 'chain' should be 2
    f1=chain[0]
    f2=chain[1]

    return (f2(f1(x)))

C1=[square,square]

x=np.array([1,2,3])
output=chain_length_2(C1,x)
print(output)
