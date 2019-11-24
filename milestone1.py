### Import the necessary libraries
import argparse
from utils import *

# Milestone 1: Median filter and Edge detection

'''
Before any further processing, a median filter is applied in order to reduce any salt and pepper noise
that may be present in the input image.
'''
# Function to apply median filter
def apply_median_filter(img, ksize):
	median = cv2.medianBlur(img, ksize)
	return median

'''
For detecting the edges, Canny edge detection technique is used. The benefits of using the Canny edge 
detector instead of a Laplacian kernel is that the edges are all single pixel edges in the resulting 
image. This allows for morphological operators to be employed more predictably on the resulting edge 
image. 
'''
# Function to detect the edges in the image

def detect_edges_filtered(img, sigma=0.33):
	# img = apply_median_filter(img, 5)
	# Compute the median of the single channel pixel intensities
	v = np.median(img)
 
	# Set lower and upper thresholds of canny using the median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(img, lower, upper)
 
	return edged

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Milestone 1 progress')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-m', '--median_kernel_size', required=False, default=7, 
        type=int, help='Increase this value to reduce no. of edges & get a more smooth output')

	args = parser.parse_args()
		
	# Read the input image and display
	img = read_input(args.input)
	display_image(img)

	# Apply median filter to remove the noise and display the images
	median_filtered_img = apply_median_filter(img, args.median_kernel_size)
	plotImages(img, median_filtered_img, 'Input image', 'Median filtered image')

	# Detect the edges in the image and display the images
	edges = detect_edges_filtered(median_filtered_img)
	plotImages(median_filtered_img, edges, 'Median filtered image', 'Edge detection')