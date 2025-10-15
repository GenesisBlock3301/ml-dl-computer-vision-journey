import argparse
import imutils
import cv2
# should go through this research paper.
from skimage.metrics import structural_similarity as compare_ssim


ap = argparse.ArgumentParser()

imageA = cv2.imread('./img/img1.png')
imageB = cv2.imread('./img/img2.png')

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# find the structural similarity index SSIM between the two.
# images, ensuring that different image is required
# Score range between [-1, 1] for perfection.
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print(f"SSIM: {score}")

# threshold the difference image, followed by finding contours to
# get the regions of the two input images that differ.
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ.
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)



cv2.imshow("Original", imageA)
cv2.imshow('Modified', imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Threshold",thresh)
cv2.waitKey(0)


