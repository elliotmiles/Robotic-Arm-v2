import cv2 as cv
import numpy as np

img = cv.imread("D:\ellio\Pictures\Camera Roll\WIN_20250830_00_13_23_Pro.jpg")

if img is None:
    raise FileNotFoundError("Could not load image")

print(img.shape)

# resize image to fit on screen
def img_resize(img, sf):
    width = int(img.shape[1] * sf)
    height = int(img.shape[0] * sf)
    dimensions = (width, height)
    return dimensions

dimensions = img_resize(img, 0.4)

disp_img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
grey = cv.cvtColor(disp_img, cv.COLOR_BGR2GRAY)

# create detector
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

# detection
corners, ids, rejected = detector.detectMarkers(grey)

ls = ids.tolist()

# print results
print("Detected markers: ", ls)
if ids is not None:
    cv.aruco.drawDetectedMarkers(disp_img, corners, ids)
    cv.imshow('Detected Markers', disp_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
