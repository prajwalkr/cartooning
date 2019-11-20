import dlib, cv2
import argparse
import numpy as np
from utils import *
from face_utils import *
from sklearn.cluster import MiniBatchKMeans


def apply_texture_face(img, num_clusters):

	img = read_input(img)
	(h, w) = img.shape[:2]

	# Convert the image to LAB color space. This is required for KMeans which is applied for clutering. 
	# The KMeans uses Euclidean distance and the Euclidean distance in LAB color space implies perceptul meaning.
	img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
	 
	# Reshape the image into a feature vector to apply k-means
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	 
	# Apply KMeans using the specified number of clusters 
	kmeans = MiniBatchKMeans(n_clusters = num_clusters)
	labels = kmeans.fit_predict(img)

	# Generate the quantized image based on the predictions
	quantized_img = kmeans.cluster_centers_.astype("uint8")[labels]
	 
	# Reshape the feature vectors to images
	quantized_img = quantized_img.reshape((h, w, 3))
	 
	# Convert from LAB to RGB
	face_img = cv2.cvtColor(quantized_img, cv2.COLOR_LAB2RGB)

	return face_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face landmarks using dlib')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-p', '--part', help='Part you want to segment', default='full')
	parser.add_argument('-c', '--cluster_size', required=False, default=4, type = int, help='Number of clusters')
	args = parser.parse_args()

	img, face, landmarks, box = get_landmarks(args.input)

	if(face is not None):
		mask = part_extractor(img, args.part)	

		img[box.top() : box.bottom(), box.left() : box.right()]

		display_image(face_img)

	else:
		print('No faces found')