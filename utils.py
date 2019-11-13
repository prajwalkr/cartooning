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

def get_points_within_contour(img, points):
	if len(img.shape) > 2: img = img[..., 0]
	cimg = np.zeros_like(img)
	cv2.drawContours(cimg, [points], 0, color=255, thickness=-1)

	display_image(cimg)
	return
	# Access the image pixels and create a 1D numpy array then add to list
	# pts = np.where(cimg == 255)
	# lst_intensities.append(img[pts[0], pts[1]])
