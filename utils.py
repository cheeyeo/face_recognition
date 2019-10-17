# Util functions for processing images
import os
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle

def get_embedding(face_pixels, model):
	face_pixels = face_pixels.astype('float32')

	# Normalize pixel values across all channels
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# Convert to channels first format
	face_pixels = np.moveaxis(face_pixels, 2, 0)

	sample = np.expand_dims(face_pixels, axis=0)

	yhat = model.predict(sample)

	return yhat[0]

def extract_face(filename, required_size=(96, 96)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = np.asarray(image)

	detector = MTCNN()
	# Note that MTCNN does not perform face alignment
	# TODO: add face alignment?
	results = detector.detect_faces(pixels)

	x1, y1, width, height = results[0]['box']

	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)

	face_array = np.asarray(image)
	return face_array


def load_faces(directory):
	faces = list()

	for filename in os.listdir(directory):
		filename = os.path.sep.join([directory, filename])
		face = extract_face(filename)
		faces.append(face)

	return faces

def load_dataset(directory):
	tmpX, tmpy = list(), list()
	for subdir in os.listdir(directory):
		path = os.path.sep.join([directory, subdir])
		print('[INFO] Path is: {}'.format(path))

		if not os.path.isdir(path):
			continue

		faces = load_faces(path)

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