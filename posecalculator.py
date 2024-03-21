import pyrealsense2 as rs
import numpy as np
import cv2
from statistics import mode
import re
import time


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.ppx
    _intrinsics.ppy = cameraInfo.ppy
    _intrinsics.fx = cameraInfo.fx
    _intrinsics.fy = cameraInfo.fy
    # _intrinsics.model = cameraInfo.distortion_mode
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.coeffs]
    print('Intrinsics are:',_intrinsics)
    print(_intrinsics.model)
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    return result

def imageinfo():
    #returns xPX,yPX and depth# This will also do indexing
    pass
    
def objectpixellocator():
    pass
    