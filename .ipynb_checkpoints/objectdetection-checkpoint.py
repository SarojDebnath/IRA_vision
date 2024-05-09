import cv2
import glob
import time
import torch
import numpy as np
from . import camera as cam
import matplotlib.pyplot as plt
#import IRA_vision
import ctypes
#directory=IRA_vision.__file__
#directory=directory[:-11]
import os
directory=os.path.dirname(os.path.abspath(__file__))
import clr
clr.AddReference(f"{directory}/IRA_UR_SocketCtrl_Prog")
import IRA_UR_SocketCtrl_Prog
WEIGHT=''
MODEL=[]

def video_scan(index=0,size=[640,480],rot=0,modelPath='',object='',condition_variable=[0,0,0,0,0,0],pose=False,robot_ip='192.168.1.251',exit_variable=[0,0,0,0,0,0],exit_operator='<',robotspeedscanning=[0,0,0,0,0,0],robotadress=False,robotid='123'):
    global WEIGHT,MODEL
    k=0
    stage=0
    count=0
    flag=True
    if pose==True:
        if robotadress==True:
            robotid=int(robotid)
            robot=ctypes.cast(robotid, ctypes.py_object).value
        else:
            robot=IRA_UR_SocketCtrl_Prog.SocketCtrl(robot_ip,30002,30020,100,1000)
            print(robot.Start())
        time.sleep(1)
        initial_position=list(robot.ActualPoseCartesianRad)
        for i in range(6):
            if condition_variable[i]!= 0.0:
                margin_variable=condition_variable[i]
                k=i
            if exit_variable[i]!=0.0:
                exit_variable_margin=exit_variable[i]
        init_pos=list(robot.ActualPoseCartesianRad)[k]
    else:
        margin_variable=condition_variable
        exit_variable_margin=exit_variable
        init_time=time.time()
    directory=IRA_vision.__file__
    directory=directory[:-11]

    if modelPath!=WEIGHT:
        model = torch.hub.load(directory, 'custom', path=modelPath, source='local')
        WEIGHT=modelPath
        MODEL=model
    else:
        model=MODEL
    cap=cv2.VideoCapture(index)
    cap.set(3,size[0])
    cap.set(4,size[1])
    t1=time.time()
    while True:
        if pose==True:
            pos1=list(robot.ActualPoseCartesianRad)
            pos1=pos1[k]
            state=abs(pos1-init_pos)
        else:
            state=abs(time.time()-init_time)
        while True:
            ret,frame=cap.read()
            if ret==True:
                height, width = frame.shape[:2]
                center = (width/2, height/2)
                rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rot, scale=1)
                frame = cv2.warpAffine(src=frame, M=rotate_matrix, dsize=(width, height))
                break
        t2=time.time()
        if t2-t1>1:
            results = model(frame)
            # Get the bounding box coordinates and labels of detected objects
            bboxes = results.xyxy[0].numpy()
            labels = results.names
            for i, bbox in enumerate(bboxes):
                # Get the coordinates of the top-left and bottom-right corners of the bounding box
                x1, y1, x2, y2 = bbox[:4].astype(int)
                confidence = round(float(bbox[4]), 2)
                if confidence>=0.75 and labels[int(bbox[5])]==object:
                    label = f"{labels[int(bbox[5])]}: {confidence}:[{x1/2+x2/2},{y1/2+y2/2}]"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if flag == True:
                        count+=1
                        flag=False
        if pose==True:
            robot.SpeedL(robotspeedscanning,False,True,0.3,0.5,0.0)
                                
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
        #Flag assignment:
        if int(state/margin_variable)!=stage:
            flag=True
            stage=int(state/margin_variable)
        #Exit Condition
        if eval(f"{state} {exit_operator} {exit_variable_margin}"):
            print('Exit condition matched')
            if pose==True:
                print(list(robot.ActualPoseCartesianRad))
                robot.MoveL(initial_position,True,True,True,0.3,0.5,0.0)
            break
    cap.release()
    cv2.destroyAllWindows()
    if robotadress==False:
        robot.Stop()
    return count
    
def localize(modelPath,object,imgPath):
    global WEIGHT,MODEL
    directory=IRA_vision.__file__
    directory=directory[:-11]
    if modelPath!=WEIGHT:
        model = torch.hub.load(directory, 'custom', path=modelPath, source='local')
        WEIGHT=modelPath
        MODEL=model
    else:
        model=MODEL
    # Load image
    image = cv2.imread(imgPath)
    # Detect objects in the image
    results = model(image)

    # Get the bounding box coordinates and labels of detected objects
    bboxes = results.xyxy[0].numpy()
    labels = results.names
    count=0
    point=[]
    for i, bbox in enumerate(bboxes):
        confidence = round(float(bbox[4]), 2)
        # Get the coordinates of the top-left and bottom-right corners of the bounding box
        x1, y1, x2, y2 = bbox[:4].astype(int)
        if labels[int(bbox[5])]==object and confidence>=0.8:
            # Draw the bounding box rectangle and label text
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(image, labels[int(bbox[5])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite('detected_image.jpg',image)
            cv2.imshow('detected',image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            x=(x1+x2)/2
            y=(y1+y2)/2
            point.append([count,x,y])
            count+=1
    number=len(point)
    return point,number
    
def click_and_localize(index,size,rot,mode,T,directory,object,name,modelpath):
    cam.image(index,size,rot,mode,T,directory,name)
    imagePath=f'{directory}/{name}'
    if type(object) is list:
        ret=[]
        #for module in object:
            #ret.append([])
        for module in object:
            point,number=localize(modelpath,module,imagePath)
            #ret[object.index(module)].append(number)
            ret.append(number)
    else:
        point,ret=localize(modelpath,object,imagePath)
    return point,ret
    
def scan_static(index,size,rot,mode,T,directory,object,name,modelpath,robot_ip,margin,iteration,robotadress,robotid):
    ret=[]
    if robotadress==True:
        robotid=int(robotid)
        robot=ctypes.cast(robotid, ctypes.py_object).value
    else:
        robot=IRA_UR_SocketCtrl_Prog.SocketCtrl(robot_ip,30002,30020,100,1000)
        print(robot.Start())
    time.sleep(1)
    initial_position=list(robot.ActualPoseCartesianRad)
    point,value=click_and_localize(index,size,rot,mode,T,directory,object,name,modelpath)
    ret.append(value)
    for i in range(iteration):
        target_pos=list(robot.ActualPoseCartesianRad)
        for j in range(6):
            target_pos[j]+=margin[j]
        robot.MoveL(target_pos,True,True,True,0.3,0.5,0.0)
        point,value=click_and_localize(index,size,rot,mode,T,directory,object,name,modelpath)
        ret.append(value)

    robot.MoveL(initial_position,True,True,True,0.3,0.5,0.0)
    if robotadress==False:
        robot.Stop()
    return ret