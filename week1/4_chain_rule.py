import numpy as np
import matplotlib.pyplot as plt


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

def sigmoid(x:np.ndarray)->np.ndarray:
    return 1/(1+np.exp(-x))

def chain_deriv_2(chain:Chain,input_range:np.ndarray)->np.ndarray:
    #(f2(f1(x)))'=f2'(f1(x))*f1'(x)
    assert len(chain)==2
    assert input_range.ndim==1 #function requires a 1 dimensional ndarray as input_range

    f1=chain[0]
    f2=chain[1]
     #f(x)
    f1_of_x=f1(input_range)

    # df1/du (u(x))
    df1dx=deriv(f1,input_range)

    # df2/du(f1(x))
    df2du=deriv(f2,f1(input_range))

    # (f2(f1(x)))' = df2du*df1dx
    return df2du*df1dx  


PLOT_RANGE= np.arange(-3,3,0.01)

chain_1=[square,sigmoid]
chain_2=[sigmoid,square]


output_chain_1=chain_length_2(chain_1,PLOT_RANGE)
output_chain_2=chain_length_2(chain_2,PLOT_RANGE)

deriv_chain_1=chain_deriv_2(chain_1,PLOT_RANGE)
deriv_chain_2=chain_deriv_2(chain_2,PLOT_RANGE)


fig1,ax1=plt.subplots(figsize=(3,3)) #n_row,n_col,plot_number
plt.title("chain1")
ax1.plot(output_chain_1,label='c1')
ax1.plot(deriv_chain_1,label='deriv_c1')
ax1.legend()

fig2,ax2=plt.subplots(figsize=(3,3)) #n_row,n_col,plot_number
plt.title("chain2")
ax2.plot(output_chain_2,label='c2')
ax2.plot(deriv_chain_2,label='deriv_c2')
ax2.legend()
plt.show()
