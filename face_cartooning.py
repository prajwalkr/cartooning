import dlib, cv2
import argparse
import numpy as np
from utils import *
from face_utils import *
from sklearn.cluster import MiniBatchKMeans

def apply_texture_face(img, num_clusters, display=False):
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
	clustered_img = kmeans.cluster_centers_.astype("uint8")[labels]
	 
	# Reshape the feature vectors to images
	clustered_img = clustered_img.reshape((h, w, 3))
	 
	# Convert from LAB to RGB
	face_img = cv2.cvtColor(clustered_img, cv2.COLOR_LAB2RGB)

	if display: display_image(face_img)

	return face_img

def toonify_face(args, input_img=None, display=False):
	img, face, landmarks, box = get_landmarks(args.input, display=display)

	if input_img is not None:
		img = input_img

	if(face is not None):
		mask = part_extractor(img, landmarks, args.part, display=display)	
		mask_crop = mask[box.top() : box.bottom(), box.left() : box.right()]

		face_img = face * mask_crop
		
		cartooned = apply_texture_face(face_img, args.cluster_size, display=display)

		blank_img_with_cartoon_face = img.copy()
		blank_img_with_cartoon_face[box.top() : box.bottom(), box.left() : box.right()] = cartooned

		img_with_cartoon_face = img * (1 - mask) + blank_img_with_cartoon_face * mask

		if display: display_image(img_with_cartoon_face)
		return img_with_cartoon_face
	else:
		return input_img

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face landmarks using dlib')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-p', '--part', help='Part you want to segment', default='full')
	parser.add_argument('-c', '--cluster_size', required=False, default=8, type = int, help='Number of clusters')
	args = parser.parse_args()

	toonify_face(args, display=True)