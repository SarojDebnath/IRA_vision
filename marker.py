import cv2
from cv2 import aruco
import numpy as np
import time

camera_matrix = np.array([[966.80083369, 0.0, 649.99730882],
                          [0.0, 971.15680184, 362.74810822],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([[0.12043679, -0.12656466, -0.00104852, 0.00258223, -0.67734902]])

class marker:
    
    def __init__(self,camIndex,size):
        
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
        self.detector_params = aruco.DetectorParameters()
        self.cap=cv2.VideoCapture(camIndex)
        self.cap.set(3,size[0])
        self.cap.set(4,size[1])
        
    def euler_from_quaternion(self,x, y, z, w):
  
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
          
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
          
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
          
        return roll_x, pitch_y, yaw_z

    def angles(self,marker_ids,tvecs,rvecs):
        for i, marker_id in enumerate(marker_ids):
            transform_translation_x = tvecs[i][0][0]
            transform_translation_y = tvecs[i][0][1]
            transform_translation_z = tvecs[i][0][2]
        
            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()   
             
            # Quaternion format     
            transform_rotation_x = quat[0] 
            transform_rotation_y = quat[1] 
            transform_rotation_z = quat[2] 
            transform_rotation_w = quat[3] 
             
            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = self.euler_from_quaternion(transform_rotation_x, transform_rotation_y, transform_rotation_z, transform_rotation_w)
             
            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = math.degrees(yaw_z)
            print("transform_translation_x: {}".format(transform_translation_x))
            print("transform_translation_y: {}".format(transform_translation_y))
            print("transform_translation_z: {}".format(transform_translation_z))
            print("roll_x: {}".format(roll_x))
            print("pitch_y: {}".format(pitch_y))
            print("yaw_z: {}".format(yaw_z))
            return [transform_translation_x,transform_translation_y,transform_translation_z,roll_x,pitch_y,yaw_z]
            
    #This function calculates the matrix required for transformation. 
    #Also, if the matrix is not available. It gives the fixed points, Also the angles.
    
    def cal_marker(self,markerlen,matrix,fixed_marker_corners,angles):
        t_start=time.time()
        while True:
            _, frame = self.cap.read()
            if _ == False and time.time()-t_start>=0.5:
                raise('No Camera')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_params)
            if time.time()-t_start>=0.5 and marker_ids is not None:
                # Draw detected markers
                aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, markerlen, camera_matrix, dist_coeffs)
                break
        cv2.imshow("Image", frame)
        cv2.waitKey(1000)
        self.cap.release()
        cv2.destroyAllWindows()
        if marker_ids is not None and matrix==False:
            return [marker_corners[0], marker_ids,rvecs, tvecs]
        if matrix==True:
            return cv2.getPerspectiveTransform(np.float32(marker_corners[0]),np.float32(fixed_marker_corners))
        if marker_ids is not None and angles==True:
            return self.angles(marker_ids,tvecs,rvecs)