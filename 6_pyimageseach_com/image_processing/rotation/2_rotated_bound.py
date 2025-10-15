import cv2
import numpy as np
import matplotlib.pyplot as plt


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image around its center without changing the dimensions.

    Parameters:
        image: np.ndarray - input image
        angle: float - rotation angle in degrees

    Returns:
        np.ndarray - rotated image
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated


def rotate_image_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image around its center and adjust bounds to fit the entire rotated image.

    Parameters:
        image: np.ndarray - input image
        angle: float - rotation angle in degrees

    Returns:
        np.ndarray - rotated image with adjusted bounds
    """
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center_x
    rotation_matrix[1, 2] += (new_height / 2) - center_y

    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated


def main():
    image_path = '../img/img1.png'
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    rotated = rotate_image(image, 45)
    rotated_bound = rotate_image_bound(image, 45)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.title("Rotated")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rotated_bound, cv2.COLOR_BGR2RGB))
    plt.title("Rotated with Bound")
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
