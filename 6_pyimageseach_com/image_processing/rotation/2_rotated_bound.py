import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate(_image, angle):
    (h, w) = _image.shape[:2]
    (_cX, _cY) = (w // 2, h // 2)
    _M = cv2.getRotationMatrix2D((_cX, _cY), -angle, 1.0)
    _image = cv2.warpAffine(_image, _M, (w, h))
    return _image


def rotate_bound(_image, _angle):
    (h, w) = _image.shape[:2]
    (_cX, _cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((_cX, _cY), -_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    _nW = int((h * sin) + (w * cos))
    _nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (_nW / 2) - _cX
    M[1, 2] += (_nH / 2) - _cY
    _image = cv2.warpAffine(image, M, (_nW, _nH))
    return _image


image = cv2.imread('../img/img1.png')

rotated_image = rotate(image, 45)
rotated_bound_image = rotate_bound(image, 45)

plt.subplot(1, 2, 1)
plt.imshow(rotated_image)
plt.subplot(1, 2, 2)
plt.imshow(rotated_bound_image)
plt.show()
