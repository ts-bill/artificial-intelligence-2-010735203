import numpy as np
import matplotlib.pyplot as plt

from random import seed
from random import random


from typing import Callable, Dict, List, Tuple

# y=w1*x1+w2*x2+w3*x3+...+wn*xn

def data_generation(x:np.ndarray)->np.ndarray:
    _w=np.array([1,5,-30,69])
    y=np.matmul(x,_w)
    return y.reshape(5,1)

def forward_loss(X_batch:np.ndarray,
                              y_batch:np.ndarray,
                              weights:Dict[str,np.ndarray])->Tuple[float,Dict[str,np.ndarray]]:
    assert X_batch.shape[0]==y_batch.shape[0]
    assert X_batch.shape[1]==weights['W'].shape[0]

    assert weights['B'].shape[0]==weights['B'].shape[1]==1

    N=np.matmul(X_batch,weights['W'])
    P=N + weights['B']

    loss=np.mean(np.power(y_batch-P,2))

    forward_info: Dict[str,np.ndarray]={}
    forward_info['X']=X_batch
    forward_info['N']=N
    forward_info['P']=P
    forward_info['y']=y_batch

    return loss, forward_info

def loss_gradients(forward_info:Dict[str,np.ndarray],
                   weights: Dict[str,np.ndarray])->Dict[str,np.ndarray]:
    batch_size=forward_info['X'].shape[0]
    

    dLdP=-2*(forward_info['y']-forward_info['P'])

    dPdN=np.ones_like(forward_info['N'])
    dPdB=np.ones_like(weights['B'])

    dLdN=dLdP*dPdN

    dNdW=np.transpose(forward_info['X'],(1,0))
    
    dLdW=np.matmul(dNdW,dLdN)


    dLdB=(dLdP*dPdB).sum(axis=0)

    loss_gradients:Dict[str,np.ndarray]={}
    loss_gradients['W']=dLdW
    loss_gradients['B']=dLdB

    return loss_gradients

#train the model
#weight initialization
weights: Dict[str,np.ndarray]={}
weights['W']=np.random.randn(4,1)
weights['B']=np.zeros((1,1))

print('init_weight:',weights['W'])

learning_rate=0.01

loss_data=[]
epoch=10000
for i in range(epoch):
    batch_size=5
    x1 = np.random.uniform(size=(batch_size))
    x2 = np.random.uniform(size=(batch_size))
    x3 = np.random.uniform(size=(batch_size))
    x4 = np.random.uniform(size=(batch_size))
    x_batch=np.transpose(np.array([x1,x2,x3,x4]))
    # print(x_batch.shape)
    # print(weights['W'].shape)
    y_batch=data_generation(x_batch)
    # print(y_batch.shape)
    # print(x_batch)
    # print(y_batch)   
    loss,forward_info=forward_loss(x_batch,y_batch,weights)
    loss_grads=loss_gradients(forward_info,weights)

    for key in weights.keys():
        weights[key]-=learning_rate*loss_grads[key]
    if(epoch%10==0):
        # print("epoch:{0},loss:{1}".format(epoch,loss))
        loss_data.append(loss)

    if(loss<1e-4):
        print("end")
        break
loss_array=np.asarray(loss_data)

print('trained_weight',+weights['W'])

plt.plot(loss_array)
plt.ylabel('loss')
plt.show()



