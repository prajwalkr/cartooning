### Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


### Function to read the input image 
def read_input(filename):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print("Image dimension:", img.shape)

	return img


# Function to display the image
def display_image(img):
	fig = plt.figure(figsize=(4, 8))
	plt.imshow(img)
	plt.show()


# Milestone 1: Median filter and Edge detection

'''
Before any further processing, a median filter is applied in order to reduce any salt and pepper noise
that may be present in the input image.
'''
# Function to apply median filter
def apply_median_filter(img):
	median = cv2.medianBlur(img, 3)
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


# Function to plot two images
def plotImages(im1, im2, t1, t2):
    
    fig=plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.title(t1)
    plt.xticks([])
    plt.yticks([])
    if (im1.ndim == 2):
        plt.imshow(im1, cmap='gray') 
    elif (im1.ndim == 3):
        plt.imshow(im1) 

    plt.subplot(1, 2, 2)
    plt.title(t2)
    plt.xticks([])
    plt.yticks([])
    if (im1.ndim == 2):
        plt.imshow(im2, cmap='gray') 
    elif (im1.ndim == 3):
        plt.imshow(im2) 
    plt.show()

    return


# Milestone 2: Morphological operations
   
'''
Dilation: Dilation is performed with a small 2 X 2 structuring element. The purpose of this step is to both bolden 
and smooth the contours of the edges slightly.
'''
def dilation(img, ksize):
    kernel = np.ones((ksize, ksize),np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations = 1)
    return dilated_img

# Read the input image and display
img = read_input('images/img.jpg')
display_image(img)

# Apply median filter to remove the noise and display the images
median_filtered_img = apply_median_filter(img)
plotImages(img, median_filtered_img, 'Input image', 'Median filtered image')

# Detect the edges in the image and display the images
edges = detect_egdes(median_filtered_img, 180, 100)
plotImages(median_filtered_img, edges, 'Median filtered image', 'Edge detection')

# Dilate the image
dilated_img = dilation(edges, 2)
plotImages(edges, dilated_img, 'Edge detection', 'Dilated image')