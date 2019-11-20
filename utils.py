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

def renormalize(img):
    m, M = img.min(), img.max()
    if M == 0: return img

    img = (255. * ((img - m).astype(np.float32) / (M - m))).astype(np.uint8)

    return img

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

def get_points_within_contour(img, points, display=False):
	if len(img.shape) > 2:
		img = img[..., 0]
		expand_dims = True

	mask = np.zeros_like(img)
	cv2.drawContours(mask, [points], 0, color=255, thickness=-1)

	if display: display_image(mask)

	if expand_dims:
		mask = np.expand_dims(mask, axis=2)

	return mask // 255