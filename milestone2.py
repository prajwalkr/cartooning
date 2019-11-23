### Import the necessary libraries
import argparse
import numpy as np
import argparse
from utils import *
from milestone1 import apply_median_filter, detect_egdes


# Milestone 2: Morphological operations
   
'''
    Dilation: Dilation is performed with a small 2 X 2 structuring element. 
    The purpose of this step is to both bolden and smooth the contours of the edges slightly.
'''
def dilation(img, ksize):
    kernel = np.ones((ksize, ksize),np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations = 1)
    return dilated_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Milestone 2 progress')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-l', '--edgemin', required=False, default=180, type=int)
    parser.add_argument('-r', '--edgemax', required=False, default=100, type=int)
    parser.add_argument('-m', '--median_kernel_size', required=False, default=3, type=int)
    parser.add_argument('-f', '--dilation_size', required=False, default=2, type=int)

    args = parser.parse_args()
        
    # Read the input image and display
    img = read_input(args.input)
    display_image(img)

    # Apply median filter to remove the noise and display the images
    median_filtered_img = apply_median_filter(img, args.median_kernel_size)
    plotImages(img, median_filtered_img, 'Input image', 'Median filtered image')

    # Detect the edges in the image and display the images
    edges = detect_egdes(median_filtered_img, args.edgemin, args.edgemax)
    plotImages(median_filtered_img, edges, 'Median filtered image', 'Edge detection')

    # Dilate the image
    dilated_img = dilation(edges, args.dilation_size)
    plotImages(edges, dilated_img, 'Edge detection', 'Dilated image')