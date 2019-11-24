import argparse
from utils import *
from milestone1 import apply_median_filter, detect_edges_filtered
from milestone2 import dilation
from milestone3_and_4 import apply_bilateral_filter
from milestone5 import quantize_colors
from milestone6 import merge_images
from cartoonize_face import toonify_face

# Combine all these functions to compute the results of step 1
def generate_edge_image(path, median_kernel_size, dilation_size):
	img = read_input(path)
	while max(img.shape) > 2048:
		img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
	# display_image(img)
	median_filtered_img = apply_median_filter(img, median_kernel_size)

	edges = detect_edges_filtered(median_filtered_img.copy())

	dilated_img = dilation(edges, dilation_size)

	return img, dilated_img

# Combine the steps to generate the results of step 2
def generate_color_image(img, median_kernel_size, quantization_factor):

	bilateral_filtered_img = apply_bilateral_filter(img)

	filtered_img = apply_median_filter(bilateral_filtered_img, median_kernel_size)

	color_img = quantize_colors(filtered_img, quantization_factor)
		
	return color_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Cartoon any given image')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-m', '--median_kernel_size', required=False, default=7, 
		type=int, help='Increase this value to reduce no. of edges & get a more smooth output')
	parser.add_argument('-f', '--dilation_size', required=False, default=2, type=int,
		help='Increase this value to get thicker edges')
	parser.add_argument('-q', '--quantization_factor', required=False, default=24, type=int,
		help='Controls cartooning effect by creating blobs, increase to get larger, unifrom blobs')

	#### Arguments for Thresholding-based improvements
	parser.add_argument('-C', '--C', required=False, help='thresholding parameter C',
							 default=3, type=int)
	parser.add_argument('-b', '--blockSize', required=False, help='thresholding parameter blockSize',
							 default=9, type=int)
	parser.add_argument('-r', '--reduce_speckles', required=False, help='reduce speckle edges',
							 default=3, type=int)

	#### Arguments for Face-specific improvements
	parser.add_argument('-u', '--unedged', required=False, 
			help='=True displays original implementation without edge overlays', default=False)
	parser.add_argument('-p', '--part', required=False, help='Face Part to add face effects', default='full')
	parser.add_argument('-c', '--cluster_size', required=False, default=8, 
						type=int, help='Decrease this to get more blobby cartoon-style faces')
	parser.add_argument('-s', '--output_image_paper', required=False, default='results/out_paper.jpg',
						type = str, help='Path to save the output of the paper implementation')
	parser.add_argument('-o', '--output_image', required=False, default='results/out.jpg', 
						type = str, help='Path to save the output of the improved implementation')
	parser.add_argument('--overlay_parts', required=False, default=[], nargs='*', 
			type=str, action='append', help='Draw lines using over face parts. Not necessary currently.')

	args = parser.parse_args()

	img, edge_img = generate_edge_image(args.input, args.median_kernel_size, args.dilation_size)

	color_img = generate_color_image(img, args.median_kernel_size, args.quantization_factor)

	basic_cartooned_output = merge_images(edge_img, color_img)

	### improved
	improved_cartoon, thresh_edge_mask = threshold_edges(img.copy(), args)
	improved_cartoon = generate_color_image(improved_cartoon, 1, args.quantization_factor)

	### Face specific improvements
	img_with_cartoon_face, full_edge_mask = toonify_face(args, improved_cartoon, display=False)
	if full_edge_mask is not None:
		if args.unedged: basic_cartooned_output = color_img
		img_with_cartoon_face_edged = img_with_cartoon_face.copy()
		for part in args.overlay_parts:
			args.part = part[0]
			_, edge_mask = toonify_face(args, improved_cartoon, display=False)

			img_with_cartoon_face_edged = merge_images(edge_mask, img_with_cartoon_face_edged)
		img_with_cartoon_face_edged = cv2.bitwise_and(img_with_cartoon_face_edged, thresh_edge_mask)
	else:
		img_with_cartoon_face_edged = img_with_cartoon_face

	# Display the input and the cartooned images
	plotImages(img, basic_cartooned_output, 'Input image', 'Cartooned image')
	# plotImages(img, improved_cartoon, 'Input image', 'Improved Cartooned image')
	plotImages(img, img_with_cartoon_face_edged, 'Input image', 'Improved Cartooned image')

	cv2.imwrite(args.output_image_paper, basic_cartooned_output[...,::-1])
	cv2.imwrite(args.output_image, img_with_cartoon_face_edged[...,::-1])