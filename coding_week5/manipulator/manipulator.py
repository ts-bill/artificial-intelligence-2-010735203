import numpy as np
DHparams=np.loadtxt('D:/INTRO2AI/NeuralNetworksAndDeepLearning/deepLearningFromScratch\coding_week4/manipulator/DH.txt',delimiter=',')

def tranformationMatrix(a,alpha,d,theta):
    row1=np.array([np.cos(np.deg2rad(theta)),-np.sin(np.deg2rad(theta)),0,a])

    row2=np.array([np.sin(np.deg2rad(theta))*np.cos(np.deg2rad(alpha)),
        np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(alpha)),
        -np.sin(np.deg2rad(alpha)),
        -np.sin(np.deg2rad(alpha))*d])

    row3=np.array([np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(alpha)),
          np.cos(np.deg2rad(theta))*np.sin(np.deg2rad(alpha)),
          np.cos(np.deg2rad(alpha)),
          np.cos(np.deg2rad(alpha))*d])
    
    tf=row1
    tf=np.append(tf,row2)
    tf=np.append(tf,row3)
    tf=np.append(tf,[0,0,0,1])
    return tf.round(decimals=4)


print("DH-parameter:")
print("joint:i\t a\t alpha\t d\t theta")
tf=[]

for i in range(0,len(DHparams)):
    print("{0}\t {1}\t {2}\t {3}\t {4}".format(i+1,DHparams[i][0],DHparams[i][1],DHparams[i][2],DHparams[i][3]))
    tf.append(tranformationMatrix(DHparams[i][0],DHparams[i][1],DHparams[i][2],DHparams[i][3]).reshape((4,4)))

eef_tf=np.identity(4)
tf_list=[]

for i in range(0,len(tf)):
    eef_tf=np.matmul(eef_tf,tf[i])
    tf_list.append(eef_tf)



print(eef_tf)
# print(tf[0])
# print(tf[0][2][3])
# tf[0][2][3]=400
# print(tf[0][2][3])


                  
    




