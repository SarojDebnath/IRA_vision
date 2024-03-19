import pyrealsense2 as rs
import numpy as np
import cv2
from statistics import mode
import re
import time

depth_values = np.zeros((480, 640))

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
    return result[0], -result[1], -result[2]

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