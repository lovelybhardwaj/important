import cv2
import numpy as np
import os

path_src = "C:/Users/soham/Downloads/image19 (4).png"
path_tgt = "C:/Users/soham/OneDrive/Desktop/paper.webp"

img = cv2.imread(path_src)

def extract_pixels(source_image, target_image, target_range):
    # Load the source image
    source = cv2.imread(source_image)
    # source = cv2.resize(source,dsize=None, fx=0.30,fy=0.30)

    # Load the target image
    target = cv2.imread(target_image)

    # Resize the target image to match the dimensions of the source image
    target = cv2.resize(target, (source.shape[1], source.shape[0]))

    # Create a mask with white pixels where the RGB values are within the target range
    lower_range = np.array(target_range[0])
    upper_range = np.array(target_range[1])
    mask = cv2.inRange(source, lower_range, upper_range)

    # Extract the pixels by copying the source image using the mask
    extracted_pixels = cv2.bitwise_and(source, source, mask=mask)

    # Resize the extracted pixels to match the dimensions of the target image
    extracted_pixels = cv2.resize(extracted_pixels, (target.shape[1], target.shape[0]))

    # Replace the corresponding pixels in the target image with the extracted pixels
    result = np.where(mask[..., None], extracted_pixels, target)

    # Save the resulting image
    return result

# Usage example
image = extract_pixels(path_src, path_tgt, [(0, 0, 0), (220,220,220)])

cv2.imshow('frame', img)
cv2.imshow('canvas',image)
cv2.waitKey()
cv2.destroyAllWindows()

directory = "C:/Users/soham/OneDrive/Desktop/Paper"
os.chdir(directory)

cv2.imwrite('paper_1.jpg', image)
