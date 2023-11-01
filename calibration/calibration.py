import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
# objp[:,0] -= 3.5
# objp[:,1] -= 5
# objp *= 0.03
# print(objp)
# exit(0)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.bmp')
images.sort()

for fname in images:
    # print(fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(img.depth())
    # exit(0)
    # Find the chess board corners
    # cv.imshow('img', img)
    # cv.waitKey()
    ret, corners = cv.findChessboardCorners(gray, (8,11), flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
    # print(ret, corners)
    # exit(0)
 # If found, add object points, image points (after refining them)
    cv.drawChessboardCorners(img, (8,11), corners, ret)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', img)
    cv.waitKey()
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (8,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 # Draw and display the corners
    
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

objp[:,0] -= 3.5
objp[:,1] -= 5
objp *= 0.03

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv.imread(images[-3])
print(images[-3])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# print(img.depth())
# exit(0)
# Find the chess board corners
# cv.imshow('img', img)
# cv.waitKey()
ret, corners = cv.findChessboardCorners(gray, (8,11), flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
# print(ret, corners)
# exit(0)
# If found, add object points, image points (after refining them)
# if ret == True:
#     objpoints.append(objp)
corners2 = cv.cornerSubPix(gray,corners, (8,11), (-1,-1), criteria)
# imgpoints.append(corners2)

ret, rvecs, tvecs, inliners = cv.solvePnPRansac(objp, corners2, mtx, dist)
print(mtx, dist, rvecs, tvecs)