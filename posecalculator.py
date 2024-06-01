import pyrealsense2 as rs
import numpy as np
import cv2
from statistics import mode
from scipy.ndimage import rotate
import re
import time
import torch
import os
import sys
directory=os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory)

depth_values = np.zeros((480, 640))
                        
MODELPATH=''
MODEL=''

def get_desired_order(points, threshold=5):
    sorted_points = sorted(points, key=lambda p: p[1], reverse=True)
    desired_order = []
    for current_point in sorted_points:
        close_neighbor = False
        for existing_point in desired_order:
          if abs(current_point[1] - existing_point[1]) <= threshold:
            close_neighbor = True
            if current_point[0]<existing_point[0]:
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
def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.ppx
    _intrinsics.ppy = cameraInfo.ppy
    _intrinsics.fx = cameraInfo.fx
    _intrinsics.fy = cameraInfo.fy
    # _intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.coeffs]
    print('Intrinsics are:',_intrinsics)
    print(_intrinsics.model)
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    # result[0]: right, result[1]: down, result[2]: forward
    return result

def click(px=10,py=10,image=False,cutoff=0.27):
    t3=time.time()
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(device_product_line)
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            print("There is a depth camera with color sensor")
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)


    # Setup the 'High Accuracy'-mode
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance_in_meters = cutoff
    clipping_distance = clipping_distance_in_meters / depth_scale
    print(clipping_distance)
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
    # Skip first frames for auto-exposure to adjust
    for x in range(5):
        pipeline.wait_for_frames()
    try:
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
                color_frame = aligned_frames.get_color_frame()
                aligned_depth_data = np.asanyarray(aligned_depth_frame.get_data())
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())


                black_color = 0
                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color, color_image)

                cv2.namedWindow('depth_cut', cv2.WINDOW_NORMAL)
                cv2.imshow('depth_cut', bg_removed)
                camera_info = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                print(camera_info)
                t4=time.time()
                if cv2.waitKey(1) and (t4-t3 >= 6) :
                    if image==True:
                        cv2.imwrite('clippedimage.jpg',bg_removed)
                    cv2.destroyAllWindows()
                    del t4
                    del t3
                    capture=bg_removed
                    for i in range(capture.shape[0]):
                        for j in range(int(capture.shape[1])):
                            depth_values[i, j] = aligned_depth_frame.get_distance(int(j), int(i))
                    break

    finally:
        if px+90 >= 640 :
            ppx=px-90
        else: ppx=px+90
        d=depth_values[py,ppx] #Taking a depth 10 pixels towards right will make the reading much more reliable (-)
        if d!=0:
            x_w, y_w, z_w = convert_depth_to_phys_coord_using_realsense(px,py, d, camera_info)
            print('Camera Points are:',x_w, y_w, z_w)
            p1=f'{x_w} {y_w} {z_w}'
            p = [float(value) for value in p1.split(' ')]
            p.append(1.0)
        else:
            depth_dict={}
            print('We must use a window')
            for k in range(py-5,py+5):
                for l in range(ppx,ppx+5):
                    depth=depth_values[k,l]
                    if depth!=0:
                        m=[k,l]
                        depth_dict[f'{m}']=round(depth, 2)
            depth_list=depth_dict.values()
            print(depth_list)
            filtered_depth=mode(depth_list)
            print(filtered_depth)
            corr_pix=list(depth_dict.keys())[list(depth_dict.values()).index(filtered_depth)] #get corresponding pixel of the distance value
            print(corr_pix)
            corr_pix=re.split(', ',corr_pix)
            pix_corr=[]
            for num in corr_pix:
                num=re.sub(r"[\([{})\]]", "", num)
                num=int(num)
                pix_corr.append(num)
            a,b=pix_corr[0],pix_corr[1]
            d1=depth_values[b,a]
            x_w, y_w, z_w = convert_depth_to_phys_coord_using_realsense(px,py, d1, camera_info)
            print('Camera Points are:',x_w, y_w, z_w)
            p1=f'{x_w} {y_w} {z_w}'
            p = [float(value) for value in p1.split(' ')]
            p.append(1.0)
        pipeline.stop()
    return p
    
def pose(km,point,level):
    p=click(px=int(point[0]),py=int(point[1]))
    dst_p=np.matmul(km,p)
    dst_p=dst_p.tolist()
    for i in range(3):
        dst_p[i]=dst_p[i]/1000+level[i]
    return dst_p
    
def posexyz(clip,size,rot,object,modelpath,boxindex,window,winOffset):
    
    global MODEL
    global MODELPATH
    position=[0,0]
    distanceZ=0
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
    config.enable_stream(rs.stream.color,size[0], size[1], rs.format.bgr8, 30)
    profile = pipeline.start(config)


    # Setup the 'High Accuracy'-mode
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    clipping_distance_in_meters = clip
    clipping_distance = clipping_distance_in_meters / depth_scale
    print(clipping_distance)
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
    for i in range(int(preset_range.max)):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        print('%02d: %s'%(i,visulpreset))
        if visulpreset == "High Accuracy":
            depth_sensor.set_option(rs.option.visual_preset, i)
            break
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
    if modelpath != MODELPATH:
        MODEL=torch.hub.load(f'{directory}', 'custom', path=modelpath, source='local')
    #write object detection code here and update position in a loop
    t_start1=time.time()
    flag=False
    while True:
        # Stores next frameset
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_data = np.asanyarray(depth_frame.get_data())
        if color_frame and time.time()-t_start1>1:
            aligned_frames = align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            points=point_cloud.calculate(aligned_depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, size[0], 3)
            verts=rotate(verts, rot, reshape=True)
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_data = np.asanyarray(aligned_depth_frame.get_data())
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
    
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            (h, w) = color_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rot, 1.0)
            color_image = cv2.warpAffine(color_image, M, (w, h))
            black_color = 0
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), black_color, color_image)
            camera_info = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            results = MODEL(color_image)
            print(results)
            # Get the bounding box coordinates and labels of detected objects
            bboxes = results.xyxy[0].numpy()
            labels = results.names
            store_points=[]
            for k, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox[:4].astype(int)
                    if labels[int(bbox[5])]==object:
                        store_points.append((x1,y1))
            desired_order = get_desired_order(store_points)
            print(desired_order)
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
                
                if confidence>=0.55 and (x1,y1)==(current_point[0],current_point[1]) and labels[int(bbox[5])]==object:
                    label = f"{labels[int(bbox[5])]}: {confidence}:[{x1/2+x2/2},{y1/2+y2/2}]"
                    object1=[x1/2+x2/2,y1/2+y2/2]
                    position=object1
                    centerx,centery=int(position[0])+winOffset[0],int(position[1])+winOffset[1]
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
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
                        campoint=convert_depth_to_phys_coord_using_realsense(position[0], position[1], distanceZ, camera_info)
                        print('campoint',campoint)
                        flag=True
                        break
                    else:
                        raise('No depth data coming in')
            if flag==True:
                break
                #############################
    cv2.namedWindow('depth_cut', cv2.WINDOW_NORMAL)
    #xyz=verts[y,x]
    cv2.imshow('depth_cut', color_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    pipeline.stop()
    
    return campoint