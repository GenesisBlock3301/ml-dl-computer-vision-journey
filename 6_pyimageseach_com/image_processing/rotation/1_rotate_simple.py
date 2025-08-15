import cv2
import imutils
import numpy as np


image = cv2.imread('../img/img1.png')
#
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate(image, angle)
    cv2.imshow('Image', rotated)
    cv2.waitKey(0)


for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow('Image', rotated)
    cv2.waitKey(0)
