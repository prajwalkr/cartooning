import argparse
from utils import *
from milestone1 import apply_median_filter, detect_egdes
from milestone2 import dilation

def remove_small_edges(edge_img):
	contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	im = np.zeros_like(edge_img)
	for contour in contours:
		if cv2.contourArea(contour) > 25:
			cv2.drawContours(im, contour, -1, (255, 255, 255), 1)

	display_image(im)

	return im


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Milestone 3')
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

	# Remove small contours in the edges
	edge_img = remove_small_edges(dilated_img)
	plotImages(img, edge_img, 'Input image', 'Edge image')
