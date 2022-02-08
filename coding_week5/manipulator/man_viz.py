from cgi import test
from sys import displayhook
import numpy as np
import cv2
from typing import Callable, Dict, List, Tuple

class tf_manipulator(object):
    def __init__(self):
        self.DHparams=np.loadtxt('./DH.txt',delimiter=',')
        print("DH-parameter:")
        print("joint:i\t a\t alpha\t d\t theta")
        self.tf=[]
        self.joints_info=[]        

        self.operation_angle=np.array([0,0,0,0])

        for i in range(0,len(self.DHparams)):
            print("{0}\t {1}\t {2}\t {3}\t {4}".format(i+1,self.DHparams[i][0],self.DHparams[i][1],self.DHparams[i][2],self.DHparams[i][3]))
            self.tf.append(self.tranformationMatrix(self.DHparams[i][0],self.DHparams[i][1],self.DHparams[i][2],self.DHparams[i][3]).reshape((4,4)))
        
    def tranformationMatrix(self,a,alpha,d,theta):
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

    def FK(self):     
        # print("FK")
        self.tf=[]
        for i in range(0,len(self.DHparams)):
            operating_angle=self.DHparams[i][3]+self.operation_angle[i]
            # print("{0}\t {1}\t {2}\t {3}\t {4}".format(i+1,self.DHparams[i][0],self.DHparams[i][1],self.DHparams[i][2],operating_angle))
            self.tf.append(self.tranformationMatrix(self.DHparams[i][0],self.DHparams[i][1],self.DHparams[i][2],operating_angle).reshape((4,4)))
        

        eef_tf=np.identity(4)
        self.joints_info=[]
        for i in range(0,len(self.tf)):
            eef_tf=np.matmul(eef_tf,self.tf[i])
            self.joints_info.append(eef_tf)
        print(eef_tf)

class visualizer(tf_manipulator):
    def __init__(self):
        super().__init__()
        self.width=400
        self.height=400

        self.displayXY=np.zeros((self.height,self.width,3))
        self.displayYZ=np.zeros_like(self.displayXY) 
    
        self.originYZ=np.array([int(self.width/2),int(self.height-self.height/5)]) #col,row
        self.originXY=np.array([int(self.width/2),int(self.height/2)])

        self.drawAxis()
        self.FK()
        self.drawManipulator()

        self.trackbar_name = ['joint 1','joint 2','joint 3']
        self.title_window="man_viz_cv"
    def drawAxis(self):       
        
        cv2.circle(self.displayYZ,tuple(self.originYZ),3,(0,255,0),-1)
        cv2.line(self.displayYZ,(0,self.originYZ[1]),(self.width,self.originYZ[1]),(255,255,255),1,lineType= cv2.LINE_AA)
        cv2.line(self.displayYZ,(int(self.width/2),0),(int(self.width/2),self.height),(255,255,255),1,lineType= cv2.LINE_AA)

        cv2.circle(self.displayXY,tuple(self.originXY),3,(0,255,0),-1)
        cv2.line(self.displayXY,(0,self.originXY[1]),(self.width,self.originXY[1]),(255,255,255),1,lineType= cv2.LINE_AA)
        cv2.line(self.displayXY,(int(self.width/2),0),(int(self.width/2),self.height),(255,255,255),1,lineType= cv2.LINE_AA)

        #draw seperated line
        # cv2.line(self.displayXY,(0,0),(self.width,0),(255,255,255),2,lineType= cv2.LINE_AA)
        #drawAxis4

        axis_x=10
        axis_y=10
        arrow_length=50

        cv2.arrowedLine(self.displayYZ,(axis_x,self.height-axis_y),(axis_x,self.height-axis_y-arrow_length),(255,0,0),2,line_type=cv2.LINE_AA) #x
        cv2.arrowedLine(self.displayYZ,(axis_x,self.height-axis_y),(axis_x+arrow_length,self.height-axis_y),(0,255,0),2,line_type=cv2.LINE_AA) #y

        cv2.arrowedLine(self.displayXY,(axis_x,axis_y),(axis_x,axis_y+arrow_length),(0,0,255),2,line_type=cv2.LINE_AA) #x
        cv2.arrowedLine(self.displayXY,(axis_x,axis_y),(axis_x+arrow_length,axis_y),(0,255,0),2,line_type=cv2.LINE_AA) #y
        
        cv2.putText(self.displayXY,"x",(axis_x+5,axis_y+arrow_length),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2,cv2.LINE_AA)
        cv2.putText(self.displayXY,"y",(axis_x+arrow_length,axis_y+5),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,cv2.LINE_AA)
        cv2.circle(self.displayXY,(axis_x,axis_y),5,(255,0,0),-1)

        cv2.putText(self.displayYZ,"z",(axis_x,self.height-axis_y-arrow_length),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2,cv2.LINE_AA)
        cv2.putText(self.displayYZ,"y",(axis_x+arrow_length,self.height-axis_y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,cv2.LINE_AA)
        cv2.circle(self.displayYZ,(axis_x,self.height-axis_y),5,(255,0,0),-1)

        self.display=np.hstack((self.displayYZ,self.displayXY)) 

    def drawManipulator(self):
        i=0
        radius=10
        colors=[(0,255,0),(0,255,0),(255,0,0),(0,0,255)]
        prev_centerYZ=self.originYZ
        prev_centerXY=self.originXY

        for joint_info in self.joints_info:
            xy_x=self.originXY[0]+joint_info[1][3]
            xy_y=self.originXY[1]+joint_info[0][3]

            yz_y=self.originYZ[0]+joint_info[1][3]
            yz_z=self.originYZ[1]-joint_info[2][3]
            centerYZ=(int(yz_y),int(yz_z))
            centerXY=(int(xy_x),int(xy_y))
            # cv2.putText(self.displayYZ,"{0}".format(i+1),center,cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1,cv2.LINE_AA)

            cv2.circle(self.displayYZ,centerYZ,radius,colors[i],-1)
            cv2.putText(self.displayYZ,"(y:{0:.3f},z:{1:.3f})".format(joint_info[1][3],joint_info[2][3]),centerYZ,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0, 255, 0),1,cv2.LINE_AA)
            cv2.circle(self.displayXY,centerXY,radius,colors[i],-1)
            cv2.putText(self.displayXY,"(x:{0:.3f},y:{1:.3f})".format(joint_info[0][3],joint_info[1][3]),centerXY,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0, 255, 0),1,cv2.LINE_AA)
            
            cv2.line(self.displayYZ,tuple(prev_centerYZ),tuple(centerYZ),colors[i],3,lineType=cv2.LINE_AA)
            cv2.line(self.displayXY,tuple(prev_centerXY),tuple(centerXY),colors[i],3,lineType=cv2.LINE_AA)

            prev_centerYZ=centerYZ
            prev_centerXY=centerXY

            radius=radius-(i*2)       
            i=i+1
        self.display=np.hstack((self.displayYZ,self.displayXY)) 
        
    
    def on_trackbar_joint1(self,val):
        self.displayXY=np.zeros((self.height,self.width,3))
        self.displayYZ=np.zeros_like(self.displayXY) 
        self.drawAxis()

        self.operation_angle[0]=val
        self.FK()
        self.drawManipulator()        
        cv2.imshow(self.title_window,self.display)
    
    def on_trackbar_joint2(self,val):
        self.displayXY=np.zeros((self.height,self.width,3))
        self.displayYZ=np.zeros_like(self.displayXY) 
        self.drawAxis()
        
        self.operation_angle[1]=val
        self.FK()
        self.drawManipulator()        
        cv2.imshow(self.title_window,self.display)
    
    def on_trackbar_joint3(self,val):
        self.displayXY=np.zeros((self.height,self.width,3))
        self.displayYZ=np.zeros_like(self.displayXY) 
        self.drawAxis()
        
        self.operation_angle[2]=val
        self.FK()
        self.drawManipulator()        
        cv2.imshow(self.title_window,self.display)
      
    def show(self):                     
        cv2.namedWindow(self.title_window)
        cv2.createTrackbar(self.trackbar_name[0], self.title_window , 0, 180, self.on_trackbar_joint1)
        cv2.createTrackbar(self.trackbar_name[1], self.title_window , 0, 90, self.on_trackbar_joint2)
        cv2.createTrackbar(self.trackbar_name[2], self.title_window , 0, 90, self.on_trackbar_joint3)
        cv2.imshow(self.title_window,self.display)
        cv2.waitKey(0)
        
def main():
    vis=visualizer()
    vis.show()




if __name__ == "__main__":
    main()