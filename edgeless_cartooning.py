import argparse
from utils import *
from milestone1 import apply_median_filter, detect_egdes
from milestone2 import dilation
from milestone3_and_4 import apply_bilateral_filter
from milestone5 import quantize_colors
from milestone6 import merge_images
from face_cartooning import toonify_face

# Combine the steps to generate the results of step 2
def generate_color_image(img, median_kernel_size, quantization_factor):

    bilateral_filtered_img = apply_bilateral_filter(img)

    filtered_img = apply_median_filter(bilateral_filtered_img, median_kernel_size)

    color_img = quantize_colors(filtered_img, quantization_factor)
        
    return color_img

def highboost(img, k):
    gauss = cv2.GaussianBlur(img, (7, 7), 0)
    # Apply Unsharp masking
    unsharp_image = renormalize(cv2.addWeighted(img, (k + 1), gauss, -k, 0))
    return unsharp_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Milestone 5 progress')
    parser.add_argument('input', help='Input image')
    parser.add_argument('-m', '--median_kernel_size', required=False, default=3, type=int)
    parser.add_argument('-k', '--boosting_scale', required=False, default=1, type=int)
    parser.add_argument('-q', '--quantization_factor', required=False, default=24, type=int)

    #### Arguments for Face-specific improvements
    parser.add_argument('-p', '--part', required=False, help='Part you want to segment', default='full')
    parser.add_argument('-c', '--cluster_size', required=False, default=8, type = int, 
                        help='Number of clusters')

    args = parser.parse_args()

    img = read_input(args.input)
    color_img = generate_color_image(img, args.median_kernel_size, args.quantization_factor)

    boosted = highboost(color_img, args.boosting_scale)

    basic_cartooned_output = boosted
    # Display the input and the cartooned image
    display_image(basic_cartooned_output)

    img_with_cartoon_face = toonify_face(args, basic_cartooned_output, display=True)