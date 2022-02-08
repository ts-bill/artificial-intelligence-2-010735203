import numpy as np

a=[1,2,3] #list
b=[4,5,6] #list

print(type(a))

# print("a+b:",a+b)

# c=np.array([1,2,3])
# d=np.array([4,5,6])

# print("c+d:",c+d)
# print("c*d:",c*d)

# print("c.shape:",c.shape)
# print('c.sum(axis=0):',c.sum(axis=0))

e=np.array([[1,2,3],[4,5,6]]) 

# #axis 0 -> row
# #axis 1 -> col
# print("e.shape:",e.shape)
# print("e:",e)
# print("e[0,0]:",e[0,0])
# print('e.sum(axis=0):',e.sum(axis=0)) 
# print('e.sum(axis=1):',e.sum(axis=1))

f=np.array([1,2,3])
print(e*f)
# print(f*e)
print(np.matmul(e,f))
# print(e.shape)
# print(f.shape)
