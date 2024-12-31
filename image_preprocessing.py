import torch
import cv2
import numpy as np
from configs import files_path as fp

# Load the old image of the townhall and the last one I took
img_1 = cv2.imread(fp.IMG_HOUSE_1)
img_2= cv2.imread(fp.IMG_HOUSE_2)
img_3 = cv2.imread(fp.IMG_HOUSE_3)

#Load the calibration parameters
intrinsics = np.load('configs/calibration_params.npz')
F = intrinsics['mtx']
dist = intrinsics['dist']

# Undistort the images
# img_1 = cv2.undistort(img_1, F, dist)
# img_2 = cv2.undistort(img_2, F, dist)
# Resize the images
img_1 = cv2.resize(img_1, (752, 480))
img_2 = cv2.resize(img_2, (752, 480))
img_3 = cv2.resize(img_3, (752, 480))



# Convert the images to grayscale           
# img_old_gray = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY)      
# img_new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)

# Save the images   
cv2.imwrite(fp.P_IMG_HOUSE_1, img_1)
cv2.imwrite(fp.P_IMG_HOUSE_2, img_2)
cv2.imwrite(fp.P_IMG_HOUSE_3, img_3)