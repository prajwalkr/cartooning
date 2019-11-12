import argparse
from utils import *
from milestone1 import apply_median_filter, detect_egdes
from milestone2 import dilation

def apply_bilateral_filter(img):
	bilateral_filtered_img = img
	for i in range(14):
	    bilateral_filtered_img = cv2.bilateralFilter( bilateral_filtered_img, 5, 9, 9 )

	return bilateral_filtered_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Milestone 3 and 4 progress')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-l', '--edgemin', required=False, default=180, type=int)
    parser.add_argument('-r', '--edgemax', required=False, default=100, type=int)
    parser.add_argument('-f', '--dilation_size', required=False, default=2, type=int)

    args = parser.parse_args()
        
    # Read the input image and display
    img = read_input(args.input)
    display_image(img)

    # Apply median filter to remove the noise and display the images
    median_filtered_img = apply_median_filter(img)
    plotImages(img, median_filtered_img, 'Input image', 'Median filtered image')

    # Detect the edges in the image and display the images
    edges = detect_egdes(median_filtered_img, args.edgemin, args.edgemax)
    plotImages(median_filtered_img, edges, 'Median filtered image', 'Edge detection')

    # Dilate the image
    dilated_img = dilation(edges, args.dilation_size)
    plotImages(edges, dilated_img, 'Edge detection', 'Dilated image')

    # Apply bilateral filter and display the images
    bilateral_filtered_img = apply_bilateral_filter(img)
    plotImages(img, bilateral_filtered_img, 'Input image', 'Bilateral filtered image')

    # Apply median filter
    filtered_img = apply_median_filter(bilateral_filtered_img)
    plotImages(bilateral_filtered_img, filtered_img, 'Bilateral filtered image', 'Median filtered image')