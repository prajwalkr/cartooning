### Import the necessary libraries
import argparse
from utils import *

# Milestone 1: Median filter and Edge detection

'''
Before any further processing, a median filter is applied in order to reduce any salt and pepper noise
that may be present in the input image.
'''
# Function to apply median filter
def apply_median_filter(img, kszie):
	median = cv2.medianBlur(img, kszie)
	return median

'''
For detecting the edges, Canny edge detection technique is used. The benefits of using the Canny edge 
detector instead of a Laplacian kernel is that the edges are all single pixel edges in the resulting 
image. This allows for morphological operators to be employed more predictably on the resulting edge 
image. 
'''
# Function to detect the edges in the image
def detect_egdes(img, edge_min, edge_max):
	edges = cv2.Canny(img, edge_min, edge_max)
	return edges

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Milestone 1 progress')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-l', '--edgemin', required=False, default=180, type=int)
	parser.add_argument('-r', '--edgemax', required=False, default=100, type=int)
	parser.add_argument('--median_kernel_size', required=False, default=3, type=int)

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