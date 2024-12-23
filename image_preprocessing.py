import torch
# from models.superglue import SuperGlue
# from models.matching import Matching
import cv2
import numpy as np


# Load the old image of the townhall and the last one I took
townhall_img_old = cv2.imread('dataset/mairiesm_1935.jpg')
townhall_img_new = cv2.imread('dataset/mairie3.jpeg')

#Load the calibration parameters
intrinsics = np.load('configs/calibration_params.npz')
F = intrinsics['mtx']
dist = intrinsics['dist']

# Undistort the images
townhall_img_new = cv2.undistort(townhall_img_new, F, dist)

# Resize the images
townhall_img_old = cv2.resize(townhall_img_old, (640, 480))
townhall_img_new = cv2.resize(townhall_img_new, (640, 480))

# Convert the images to grayscale           
townhall_img_old_gray = cv2.cvtColor(townhall_img_old, cv2.COLOR_BGR2GRAY)      
townhall_img_new_gray = cv2.cvtColor(townhall_img_new, cv2.COLOR_BGR2GRAY)

