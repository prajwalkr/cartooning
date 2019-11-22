import dlib, cv2
import argparse
import numpy as np
from utils import *
from sklearn.cluster import MiniBatchKMeans


detector, predictor = None, None 

def init(ckpt='data/shape_predictor_81_face_landmarks.dat'):
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
	coords = np.zeros((81, 2), dtype=dtype)
 
	# loop over the 81 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 81):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords


def get_landmarks(img_or_file, display=False):
	if type(img_or_file) == str:
		img = dlib.load_rgb_image(img_or_file)
	else:
		img = img_or_file

	if not detector:
		init()

	boxes = detector(img, 0)
	if len(boxes) == 0:
		print("The image doesn't have any face!!!")
		return None, None, None, None
	box = get_max_area(boxes)

	shape = predictor(img, box)
	shape = shape_to_np(shape)

	output = img.copy()

	top = np.inf
	for coord in shape:
		top = min(top, coord[1])

	bottom = -np.inf
	for coord in shape:
		bottom = max(bottom, coord[1])

	box = dlib.rectangle(left=box.left(), top=top, right=box.right(), bottom=bottom)
	cv2.rectangle(output, (box.left(), box.top()), (box.right(), box.bottom()), (0, 255, 0), 2)

	for (x, y) in shape:
		cv2.circle(output, (x, y), 1, (0, 0, 255), -1)

	if display: plotImages(img, output, 'Input image', 'Face with landmarks')

	crop = crop_box(img, box)
	return img, crop, shape, box

def part_extractor(img, landmarks, name='full', display=False):
	if name == 'full':
		basic = landmarks[:17]
		forehead_indices = [78, 74, 79, 73, 72, 80, 71,
							70, 69, 68, 76, 75, 77]

		forehead = np.concatenate([np.expand_dims(landmarks[f], axis=0) for f in forehead_indices])

		ordered = np.concatenate([basic, forehead], axis=0)

		mask = get_points_within_contour(img, ordered, display)

	elif name == 'lip':
		outline_points = landmarks[48:]
		mask = get_points_within_contour(img, outline_points, display)

	elif name == 'lefteye':
		outline_points = landmarks[36:42]
		mask = get_points_within_contour(img, outline_points, display)

	elif name == 'righteye':
		outline_points = landmarks[42:48]
		mask = get_points_within_contour(img, outline_points, display)
		
	else:
		raise NotImplementedError('Mask type: {} not implemented yet!'.format(name))

	return mask

def crop_box(img, box):
	crop = img[max(box.top(), 0) : box.bottom(), max(box.left(), 0) : box.right()]
	return crop

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Face landmarks using dlib')
	parser.add_argument('input', help='Input image')
	parser.add_argument('-p', '--part', help='Part you want to segment', default='full')
	args = parser.parse_args()

	img, face, landmarks, box = get_landmarks(args.input, display=True)

	if(face is not None):
		mask = part_extractor(img, args.part, display=True)	
	else:
		print('No faces found')