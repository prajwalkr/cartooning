import argparse
from utils import *
from milestone1 import apply_median_filter, detect_egdes
from milestone2 import dilation
from milestone3 import remove_small_edges
from milestone4 import apply_bilateral_filter
from milestone5 import quantize_colors

def merge_images(edge_img, color_img):
	merged_img = color_img.copy()
	merged_img[edge_img == 255] = 0  
	return merged_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Milestone 5 progress')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-l', '--edgemin', required=False, default=180, type=int)
    parser.add_argument('-r', '--edgemax', required=False, default=100, type=int)
    parser.add_argument('--median_kernel_size', required=False, default=3, type=int)
    parser.add_argument('-f', '--dilation_size', required=False, default=2, type=int)
    parser.add_argument('--quantization_factor', required=False, default=24, type=int)

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

    # Apply bilateral filter and display the images
    bilateral_filtered_img = apply_bilateral_filter(img)
    plotImages(img, bilateral_filtered_img, 'Input image', 'Bilateral filtered image')

    # Apply median filter
    filtered_img = apply_median_filter(bilateral_filtered_img, args.median_kernel_size)
    plotImages(bilateral_filtered_img, filtered_img, 'Bilateral filtered image', 'Median filtered image')

    # Apply qunatization
    quantized_img = quantize_colors(filtered_img, args.quantization_factor)
    plotImages(filtered_img, quantized_img, 'Median filtered image', 'Quantized image')

    # Merge the edge image and the color image
    color_img = quantized_img
    merged_img = merge_images(edge_img, color_img)

    # Disply the merged image i.e. cartooned image
    plotImages(img, merged_img, 'Input image', 'Cartooned image')
