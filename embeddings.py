import numpy as np
from PIL import Image
from model import face_model, facenet_keras_model
from fr_utils import *
from utils import get_embedding
from keras import backend as K
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-cf", "--channels_first", action="store_true", help="Turn on/off channel first for keras image data format")
ap.add_argument("-m", "--model", type=str, required=True, help="Select model to run.")
ap.add_argument("-s", "--scale", default="normalize", required=True, help="Mode for scaling pixel values. Options are standardize or normalize")
args = vars(ap.parse_args())

if args["channels_first"]:
	K.set_image_data_format('channels_first')
else:
	K.set_image_data_format("channels_last")

print(args)

print("[INFO] Channel Ordering: {}".format(K.image_data_format()))

data = np.load('faces_dataset.npz')

trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("[INFO] Loaded dataset: trainX: {}, trainY: {}, testX: {}, testY: {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))

if args["model"] == "facenet":
	print("[INFO] Loading Keras Facenet model...")
	model = facenet_keras_model("model/facenet_keras.h5")
	model.summary()
else:
	model = face_model((3, 96, 96))
	model.summary()
	print('[INFO] Loading model weights...')
	load_weights_from_FaceNet(model)

# Convert each face image into embedding
print("[INFO] Converting faces into embedding...")
newTrainX = list()

for face_pixels in trainX:
	embedding = get_embedding(face_pixels, model, args["channels_first"], mode=args["scale"])
	newTrainX.append(embedding)

newTrainX = np.asarray(newTrainX)
print("[INFO] newTrainX shape: {}".format(newTrainX.shape))

newTestX = list()

for face_pixels in testX:
	embedding = get_embedding(face_pixels, model, args["channels_first"], mode=args["scale"])
	newTestX.append(embedding)

newTestX = np.asarray(newTestX)
print("[INFO] newTestX shape: {}".format(newTestX.shape))

np.savez_compressed('faces_embeddings.npz', newTrainX, trainY, newTestX, testY)