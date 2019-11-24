import dlib, cv2
import argparse
import numpy as np
from utils import *
from face_utils import *

# cluster face pixels into num_clusters to get smooth textured cartoon-effect
def apply_texture_face(img, num_clusters, display=False):
	clustered_img = KMeansClustering(img, num_clusters)

	if display: display_image(clustered_img)

	return clustered_img

# main driver function to cartoonify face 
def toonify_face(args, input_img=None, display=False):
	# get face crop and landmarks
	img, face, landmarks, box = get_landmarks(args.input, display=display)

	if input_img is not None:
		img = input_img

	if(face is not None):
		# we need a face mask to apply face-specific cartoon effect
		mask, edge_mask = part_extractor(img, landmarks, args.part, display=display)

		mask_crop = crop_box(mask, box)

		face_img = face * mask_crop
		cartooned = apply_texture_face(face_img, args.cluster_size, display=display)

		blank_img_with_cartoon_face = img.copy()
		blank_img_with_cartoon_face[max(box.top(), 0) : box.bottom(), 
								max(box.left(), 0) : box.right()] = cartooned

		# chroma-keying style merging background and foreground face
		img_with_cartoon_face = img * (1 - mask) + blank_img_with_cartoon_face * mask

		if display: display_image(img_with_cartoon_face)
		return img_with_cartoon_face, edge_mask
	else:
		return input_img, None

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face-specific cartooning methods')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-p', '--part', required=False, help='Face Part to add face effects', default='full')
	parser.add_argument('-c', '--cluster_size', required=False, default=8, 
		type = int, help='Decrease this to get more blobby cartoon-style faces')
	args = parser.parse_args()

	cartooned_img = toonify_face(args, display=True)

	plotImages(args.input, cartooned_img, 'Input image', 'Cartooned image')