import numpy as np
from PIL import Image
from model import face_model
from fr_utils import *
from utils import get_embedding
from keras import backend as K
import os

K.set_image_data_format('channels_first')

data = np.load('faces_dataset.npz')

trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("[INFO] Loaded dataset: trainX: {}, trainY: {}, testX: {}, testY: {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))

model = face_model((3, 96, 96))
model.summary()

print('[INFO] Loading model weights...')
load_weights_from_FaceNet(model)

if not os.path.exists("face_model.h5"):
	model.save("face_model.h5")

# Convert each face image into embedding
print("[INFO] Converting into embedding...")
newTrainX = list()

for face_pixels in trainX:
	embedding = get_embedding(face_pixels, model)
	print(embedding.shape)
	newTrainX.append(embedding)

newTrainX = np.asarray(newTrainX)
print("[INFO] newTrainX shape: {}".format(newTrainX.shape))

newTestX = list()

for face_pixels in testX:
	embedding = get_embedding(face_pixels, model)
	print(embedding.shape)
	newTestX.append(embedding)

newTestX = np.asarray(newTestX)
print("[INFO] newTestX shape: {}".format(newTestX.shape))

np.savez_compressed('faces_embeddings.npz', newTrainX, trainY, newTestX, testY)