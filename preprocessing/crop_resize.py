import cv2
import numpy as np
import os

def crop_silhouette(image):
    coords = cv2.findNonZero(image)  # Find white pixels
    if coords is None:
        return image  # No silhouette detected
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped

def resize_silhouette(image, save_path, target_size=(512, 512)):
    h, w = image.shape
    # Resize while maintaining proportion
    aspect_ratio = w / h
    if aspect_ratio > 1:  # Wider image
        new_w = target_size[0]
        new_h = int(target_size[0] / aspect_ratio)
    else:  # Higher image
        new_h = target_size[1]
        new_w = int(target_size[1] * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create square black canvas
    square_image = np.zeros(target_size, dtype=np.uint8)

    # Center resized image on canvas
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    square_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    cv2.imwrite(save_path, square_image)
    print(f"Cropped and Resized silhouette saved: {save_path}")

    return square_image