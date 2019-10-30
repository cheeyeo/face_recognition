# Util functions for processing images
import os
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
import cv2

def get_embedding(face_pixels, model, channel_first, mode='normalize'):
	face_pixels = face_pixels.astype('float32')

	if mode == 'normalize':
		# Normalize pixel values to be in range 0-1
		face_pixels /= 255.0
		print('Min pixels: {:.3f}, Max pixels: {:.3f}'.format(face_pixels.min(), face_pixels.max()))
	elif mode == 'standardize':
		# Standardize pixel values across all channels
		# Mean of 0, std dev of 1
		# Below implements global standardization
		mean, std = face_pixels.mean(), face_pixels.std()
		face_pixels = (face_pixels - mean) / std

	# Convert to channels first format
	if channel_first:
		face_pixels = np.moveaxis(face_pixels, 2, 0)

	sample = np.expand_dims(face_pixels, axis=0)

	yhat = model.predict(sample)

	return yhat[0]

def align_face(image, detector, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
	"""
	[{'box': [315, 88, 170, 222], 'confidence': 0.9803563356399536, 'keypoints': {'left_eye': (382, 164), 'right_eye': (455, 189), 'nose': (409, 217), 'mouth_left': (358, 253), 'mouth_right': (420, 272)}}]
	"""

	if desired_face_height is None:
		desired_face_height = desired_face_width

	res = detector.detect_faces(image)[0]

	# compute the center of mass for each eye
	# the keypoints returned by MTCNN already sets the center
	left_eye_center = res['keypoints']['left_eye']
	right_eye_center = res['keypoints']['right_eye']

	# compute the angle between the eye centroids
	dY = right_eye_center[1] - left_eye_center[1]
	dX = right_eye_center[0] - left_eye_center[0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180

	# compute the desired right eye x-coordinate based on the
	# desired x-coordinate of the left eye
	desired_right_eyeX = 1.0 - desired_left_eye[0]

	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
	dist = np.sqrt((dX ** 2) + (dY ** 2))
	desiredDist = (desired_right_eyeX - desired_left_eye[0])
	desiredDist *= desired_face_width
	scale = desiredDist / dist

	# compute center (x, y)-coordinates (i.e., the median point)
	# between the two eyes in the input image
	eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
		(left_eye_center[1] + right_eye_center[1]) // 2)

	# grab the rotation matrix for rotating and scaling the face
	M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

	# update the translation component of the matrix
	tX = desired_face_width * 0.5
	tY = desired_face_height * desired_left_eye[1]
	M[0, 2] += (tX - eyesCenter[0])
	M[1, 2] += (tY - eyesCenter[1])

	# apply the affine transformation
	(w, h) = (desired_face_width, desired_face_height)
	output = cv2.warpAffine(image, M, (w, h),
		flags=cv2.INTER_CUBIC)

	# return the aligned face
	return output

# Face alignment 
# Adapted from https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py
def extract_face(filename, detector, required_size=(96, 96), align=False):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = np.asarray(image)

	if align:
		print('[INFO] Performing face alignment...')
		aligned = align_face(pixels, detector, desired_face_width=256,desired_left_eye=(0.55, 0.55))
		results = detector.detect_faces(aligned)
		print("[INFO] Filename: {}, Results: {}".format(filename, results))
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# Gets the face pixels from aligned image
		face = aligned[y1:y2, x1:x2]
	else:
		# Note that MTCNN does not perform face alignment
		results = detector.detect_faces(pixels)
		x1, y1, width, height = results[0]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# Gets the face pixels from the original image
		face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)

	face_array = np.asarray(image)
	return face_array


def load_faces(directory, size, align=False):
	detector = MTCNN()
	faces = list()

	for filename in os.listdir(directory):
		filename = os.path.sep.join([directory, filename])
		face = extract_face(filename, detector, required_size=(size, size), align=align)
		faces.append(face)

	return faces

def load_dataset(directory, size, align=False):
	tmpX, tmpy = list(), list()

	for subdir in os.listdir(directory):
		path = os.path.sep.join([directory, subdir])
		print('[INFO] Path is: {}'.format(path))

		if not os.path.isdir(path):
			continue

		faces = load_faces(path, int(size), align=align)

		labels = [subdir for _ in range(len(faces))]

		print('[INFO] Loaded {:d} examples for class {}'.format(len(faces), subdir))

		tmpX.extend(faces)
		tmpy.extend(labels)

	tmpX = np.asarray(tmpX)
	tmpy = np.asarray(tmpy)

	return tmpX, tmpy

def save_obj(fname, obj):
	with open(fname, "wb") as f:
		f.write(pickle.dumps(obj))

def load_obj(fname):
	with open(fname, "rb") as f:
		return pickle.load(f)