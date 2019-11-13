import dlib, cv2
import argparse
import numpy as np
from utils import *

detector, predictor = None, None 

def init(ckpt='data/shape_predictor_68_face_landmarks.dat'):
	global detector, predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(ckpt)

def get_max_area(boxes):
	max_area, max_box = -np.inf, None
	for b in boxes:
		w = b.right() - b.left()
		h = b.bottom() - b.top()
		
		area = w * h
		if area > max_area:
			max_area = area
			max_box = b
	return max_box	

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords


def get_landmarks(img_or_file, display=True):
	if type(img_or_file) == str:
		img = dlib.load_rgb_image(img_or_file)
	else:
		img = img_or_file

	if not detector:
		init()

	boxes = detector(img, 0)
	if len(boxes) == 0:
		return None, None
	box = get_max_area(boxes)

	shape = predictor(img, box)
	shape = shape_to_np(shape)

	output = img.copy()
	cv2.rectangle(output, (box.left(), box.top()), (box.right(), box.bottom()), (0, 255, 0), 2)

	for (x, y) in shape:
		cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

	plotImages(img, output, 'Input image', 'Face with landmarks')

	crop = img[box.top() : box.bottom(), box.left() : box.right()]
	return img, crop, shape, box

def part_extractor(name='full'):
	assert name == 'full'

	mask = get_points_within_contour(img, landmarks[:27])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face landmarks using dlib')
	parser.add_argument('input', help='Input image')

	args = parser.parse_args()

	img, face, landmarks, box = get_landmarks(args.input)

	part_extractor()	