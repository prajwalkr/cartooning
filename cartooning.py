import argparse
from utils import *
from milestone1 import apply_median_filter, detect_egdes
from milestone2 import dilation
from milestone3_and_4 import apply_bilateral_filter
from milestone5 import quantize_colors
from milestone6 import merge_images
from face_cartooning import toonify_face

# Combine all these functions to compute the results of step 1
def generate_edge_image(path, median_kernel_size, edge_min, edge_max, dilation_size):
    img = read_input(path)

    median_filtered_img = apply_median_filter(img, median_kernel_size)

    edges = detect_egdes(median_filtered_img, edge_min, edge_max)

    edge_img = dilation(edges, dilation_size)

    return img, edge_img

# Combine the steps to generate the results of step 2
def generate_color_image(img, median_kernel_size, quantization_factor):

    bilateral_filtered_img = apply_bilateral_filter(img)

    filtered_img = apply_median_filter(bilateral_filtered_img, median_kernel_size)

    color_img = quantize_colors(filtered_img, quantization_factor)
        
    return color_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Milestone 5 progress')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-l', '--edgemin', required=False, default=180, type=int)
    parser.add_argument('-r', '--edgemax', required=False, default=100, type=int)
    parser.add_argument('-m', '--median_kernel_size', required=False, default=3, type=int)
    parser.add_argument('-f', '--dilation_size', required=False, default=2, type=int)
    parser.add_argument('-q', '--quantization_factor', required=False, default=24, type=int)

    #### Arguments for Face-specific improvements
    parser.add_argument('-p', '--part', required=False, help='Part you want to segment', default='full')
    parser.add_argument('-c', '--cluster_size', required=False, default=4, type = int, help='Number of clusters')

    args = parser.parse_args()

    img, edge_img = generate_edge_image(args.input, args.median_kernel_size, args.edgemin, args.edgemax, args.dilation_size)

    color_img = generate_color_image(img, args.median_kernel_size, args.quantization_factor)

    basic_cartooned_output = merge_images(edge_img, color_img)

    # Display the input and the cartooned image
    plotImages(img, basic_cartooned_output, 'Input image', 'Output (Cartooned) image before Face processing')

    toonify_face(args, basic_cartooned_output, display=True)