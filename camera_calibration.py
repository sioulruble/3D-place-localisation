# Camera calibration script


import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from configs import files_path as fp
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
 

    # parameters of the camera calibration pattern
    pattern_num_rows = 9
    pattern_num_cols = 6
    pattern_size= (pattern_num_rows, pattern_num_cols)
    #mobile phone cameras can have a very high resolution.
    # It can be reduced to reduce the computing overhead
    image_downsize_factor = 4

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_num_rows*pattern_num_cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_num_rows,0:pattern_num_cols].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(fp.CALIBRATION_IMG_PATH)
    print(f"Found images: {images}")

    #cv.namedWindow('img', cv.WINDOW_NORMAL)  # Create window with freedom of dimensions
    #cv.resizeWindow('img', 800, 600)
    for fname in images:
        img = cv.imread(fname)
        img_rows = img.shape[1]
        img_cols = img.shape[0]
        new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
        img = cv.resize(img, new_img_size, interpolation = cv.INTER_CUBIC)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        
        print('Processing caliration image:', fname)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #initial_distortion = np.zeros((1, 5))
    #initial_K = np.eye(3)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=(cv.CALIB_ZERO_TANGENT_DIST))

    # reprojection error for the calibration images
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    print('The calibartion matrix is')
    print(mtx)
    print('The radial distortion parameters are')
    print(dist)

    # undistorting the images
    print('Undistoring the images')
    images = glob.glob(fp.NEW_IMG_PATH)
    for fname in images:
        img = cv.imread(fname)
        img_rows = img.shape[1]
        img_cols = img.shape[0]
        new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
        img = cv.resize(img,new_img_size, interpolation = cv.INTER_CUBIC)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        print(fname)
        undist_image = cv.undistort(img, mtx, dist)
        cv.imshow(fname, undist_image)

    cv.waitKey(0)
    
    #write the calibration parameters to a file
    np.savez('calibration_params', mtx=mtx, dist=dist)