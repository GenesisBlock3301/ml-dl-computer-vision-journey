import cv2

img = cv2.imread('./img.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(threshold, 1, 2)

output = img.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

cv2.imshow('threshold', threshold)
cv2.imshow('Contours', output)
cv2.waitKey(0)
cv2.destroyAllWindows()