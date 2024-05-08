from simple_pid import PID
import ctypes
import cv2
import torch
import time
import pyrealsense2 as rs
import numpy as np
from statistics import mode
import math
import IRA_vision
directory=IRA_vision.__file__
directory=directory[:-11]
import clr
clr.AddReference(f"{directory}/IRA_UR_SocketCtrl_Prog")
import IRA_UR_SocketCtrl_Prog
vel_dict={0:'vel_x',1:'vel_y',2:'vel_z'}


def load_model():
    pass

class detection:
    def __init__(self,X,Y,Z,kinematics,Setpoints,ip,robotadress,robotID,Timeout,deformation,M):
        self.deformation=deformation
        self.M=M
        self.Timeout=Timeout
        self.Setpoints=Setpoints
        self.robotadress=robotadress
        self.pidx = PID(X[0], X[1], X[2], setpoint=self.Setpoints[0])
        self.pidy=PID(Y[0], Y[1], Y[2], setpoint=self.Setpoints[1])
        self.pidz=PID(Z[0], Z[1], Z[2], setpoint=self.Setpoints[2])
        self.velocity=[vel_dict[kinematics[0]],vel_dict[kinematics[1]],vel_dict[kinematics[2]]]
        #pid.setpoint = 320
        self.pidx.sample_time = 0.01
        self.pidy.sample_time = 0.01
        if self.robotadress==True:
            robotid=int(robotID)
            self.robot=ctypes.cast(robotid, ctypes.py_object).value
        else:
            self.robot=IRA_UR_SocketCtrl_Prog.SocketCtrl(ip,30002,30020,100,1000)
            print(self.robot.Start())

    def get_desired_order(self,points, threshold=10):
        sorted_points = sorted(points, key=lambda p: p[0], reverse=True)
        desired_order = []
        for current_point in sorted_points:
            close_neighbor = False
            for existing_point in desired_order:
              if abs(current_point[0] - existing_point[0]) <= threshold:
                close_neighbor = True
                if current_point[1]<existing_point[1]:
                    index = desired_order.index(existing_point) 
                    desired_order.insert(index, current_point)
                    break
                else:
                    index = desired_order.index(existing_point) 
                    desired_order.insert(index+1, current_point)
                    break 
    
            if not close_neighbor:
              desired_order.append(current_point)
            
        return desired_order
            
    def xyz(self,clip_distance,size,weights,object,boxindex,window,winOffset,z=True):
        
        time.sleep(1)
        position=[0,0]
        distanceZ=0
        print('User input:',object)
        # Create a pipeline
        pipeline = rs.pipeline()
        # Create a config and configure the pipeline to stream
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                print("There is a depth camera with color sensor")
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
        profile = pipeline.start(config)
        time.sleep(0.5)
    
        # Setup the 'High Accuracy'-mode
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = clip_distance
        clipping_distance = clipping_distance_in_meters / depth_scale
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            print('%02d: %s'%(i,visulpreset))
            if visulpreset == "High Accuracy":
                depth_sensor.set_option(rs.option.visual_preset, i)
        # enable higher laser-power for better detection
        depth_sensor.set_option(rs.option.laser_power, 180)
        # lower the depth unit for better accuracy and shorter distance covered
        depth_sensor.set_option(rs.option.depth_units, 0.0005)
        align_to = rs.stream.color
        align = rs.align(align_to)
        point_cloud = rs.pointcloud()
        # Skip first frames for auto-exposure to adjust
        for x in range(5):
            pipeline.wait_for_frames()
        model = torch.hub.load(f'{directory}', 'custom', path=f'{weights}', source='local')
        #write object detection code here and update position in a loop
        print(self.Setpoints)
        t1=time.time()
        while True:
            # Stores next frameset
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            depth_data = np.asanyarray(depth_frame.get_data())
            if color_frame:
                aligned_frames = align.process(frames)
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                points=point_cloud.calculate(aligned_depth_frame)
                verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, size[0], 3)
                ##############TRANSFORM#######
                if self.deformation==True:
                    verts = cv2.warpPerspective(verts,self.M,(verts.shape[1],verts.shape[0]))
                
                color_frame = aligned_frames.get_color_frame()
                aligned_depth_data = np.asanyarray(aligned_depth_frame.get_data())
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
        
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                if self.deformation==True:
                    color_image = cv2.warpPerspective(color_image,self.M,(color_image.shape[1],color_image.shape[0]))
        
                black_color = 0
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color, color_image)
                results = model(color_image)
                # Get the bounding box coordinates and labels of detected objects
                bboxes = results.xyxy[0].numpy()
                labels = results.names
                store_points=[]
                for k, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox[:4].astype(int)
                    if labels[int(bbox[5])]==object:
                        store_points.append((x1,y1))
                init_box=boxindex
                desired_order = self.get_desired_order(store_points)
                while True:
                    try:
                        current_point=desired_order[boxindex]
                        break
                    except IndexError:
                        if len(desired_order)>1:
                            boxindex-=1
                        else:
                            boxindex=0
                
                
                for i, bbox in enumerate(bboxes):
                    # Get the coordinates of the top-left and bottom-right corners of the bounding box
                    x1, y1, x2, y2 = bbox[:4].astype(int)
                    confidence = round(float(bbox[4]), 2)
                    
                    if confidence>=0.80 and labels[int(bbox[5])]==object and (x1,y1)==(current_point[0],current_point[1]):
                        label = f"{labels[int(bbox[5])]}: {confidence}:[{x1/2+x2/2},{y1/2+y2/2}]"
                        object1=[x1/2+x2/2,y1/2+y2/2]
                        position=object1
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        ###################################EVA SPecific#################################
                        globals()['vel_x']=self.pidx(position[0])
                        globals()['vel_y']=self.pidy(position[1])
                        globals()['vel_z']=0
                        speed=[globals()[self.velocity[0]],globals()[self.velocity[1]],globals()[self.velocity[2]],0,0,0]
                        centerx=int(position[0])+winOffset[0]
                        centery=int(position[1])+winOffset[1]
                        distance=[]
                        distanceZ=0.0
                        cv2.rectangle(color_image, (centerx-window[0], centery-window[1]), (centerx+window[0], centery+window[1]), (255, 0, 0), 1)
                        for k in range(centery-window[1],centery+window[1]):
                            for l in range(centerx-window[0],centerx+window[0]):
                                depth=verts[k,l][2]
                                #print(depth)
                                if depth!=0:
                                    distance.append(round(depth,3))
                        if distance!=[]:
                            distanceZ=mode(distance)
                        if distanceZ!=0.0:
                            if z==True:
                                globals()['vel_z']=self.pidz(distanceZ)
                                speed=[globals()[self.velocity[0]],globals()[self.velocity[1]],globals()[self.velocity[2]],0,0,0]
                                self.robot.SpeedL(speed,False,True,0.3,0.5,0.0)
                            else:
                                self.robot.SpeedL(speed,False,True,0.3,0.5,0.0)
                        elif z==False:
                            self.robot.SpeedL(speed,False,True,0.3,0.5,0.0)
                        else:
                            print('No distance data')
                            self.robot.SpeedL(speed,False,True,0.3,0.5,0.0)
                    else:
                        text=f'{labels[int(bbox[5])]}:{confidence}'
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(color_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                timeout=time.time()-t1
                cv2.namedWindow('depth_cut', cv2.WINDOW_NORMAL)
                cv2.imshow('depth_cut', color_image)
                boxindex=init_box
                print(position[0],position[1],distanceZ)
                if z==False:
                    distanceZ=self.Setpoints[2]
                if (position[0]==self.Setpoints[0]) and (position[1]==self.Setpoints[1]) and (self.Setpoints[2]-0.0005<=round(distanceZ,4)<=self.Setpoints[2]+0.0005) or timeout>=self.Timeout:
                    vel_x=self.pidx(position[0])
                    vel_y=self.pidy(position[1])
                    vel_z=self.pidz(distanceZ)
                    speed=[globals()[self.velocity[0]],globals()[self.velocity[1]],globals()[self.velocity[2]],0,0,0]
                    print('Breaking Speed',speed)
                    self.robot.SpeedL(speed,False,True,0.3,0.5,0.0)
                    self.robot.StopL(20.0)
                    camera_pose=list(self.robot.ActualPoseCartesianRad)
                    print('Time executed: ',timeout)
                    break
                if cv2.waitKey(1) & 0xFF==27:
                    break
        pipeline.stop()
        cv2.destroyAllWindows()
        print('End Position',list(self.robot.ActualPoseCartesianRad))
        if self.robotadress==False:
            self.robot.Stop()
        return camera_pose