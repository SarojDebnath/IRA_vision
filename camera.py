import cv2
import numpy as np
import time

def rot_image(image,deg):
    height, width = image.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=deg, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return rotated_image

def image(index=0,size=[640,480],rot=0,mode='rgb',T=1,directory='',name='captured.jpg'):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    t1=time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        t2=time.time()
        if t2-t1 >= T:
            frame=rot_image(frame,rot)
            if mode == 'gray':
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f'{directory}/{name}', grayFrame)
                break
            elif mode=='rgb':
                cv2.imwrite(f'{directory}/{name}', frame)
                break    
    cap.release()
    
def video(index=0,size=[640,480]):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    while True:
        ret, frame = cap.read()
        cv2.imshow('video',frame)
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        if key==27:
            break
    cap.release()
    cv2.destroyAllWindows()