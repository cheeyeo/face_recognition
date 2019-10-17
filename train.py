import os
import numpy as np
from PIL import Image
from utils import get_embedding, extract_face, save_obj
from model import face_model, triplet_loss
from fr_utils import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from keras import backend as K
from keras.models import load_model

K.set_image_data_format('channels_first')

# Face verification
def verify(image_path, target, model):
	"""
	image_path: path of source image
	target: target embedding to compare againsr
	"""
	face_pixels = extract_face(image_path)
	encoding = get_embedding(face_pixels, model)

	dist = np.linalg.norm(target - encoding)
	print(dist)

	if dist < 0.7:
		print('Its correct!')
	else:
		print('Go away!')

# Face recognition
# Compares against all other faces i.e. 1:M
def recognition(image_path, model):
	"""
	image_path: path of source image
	target: target embedding to compare against
	"""
	face_pixels = extract_face(image_path)
	encoding = get_embedding(face_pixels, model)

	min_dist = 100

	# Loop through all registered users in system
	for idx, enc in enumerate(trainX):
		name = trainY[idx]
		dist = np.linalg.norm(enc - encoding)

		if dist < min_dist:
			min_dist = dist
			identity = name

	if min_dist > 0.7:
		print("Not in database.")
		print("Dist is: {}".format(min_dist))
	else:
		print("It's {}, the distance is {}".format(identity, min_dist))


data = np.load("faces_embeddings.npz")
trainX, trainY, testX, testY = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

print("[INFO] Loaded embeddings shape: trainX: {}, trainY: {}, testX: {}, testY: {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))

# Here we are testing the embeddings by comparing distances between embeddings...

if os.path.exists("face_model.h5"):
	model = load_model("face_model.h5", custom_objects={"triplet_loss": triplet_loss})
	# model.summary()
else:
	model = face_model((3, 96, 96))
	model.summary()
	print('[INFO] Loading model weights...')
	load_weights_from_FaceNet(model)


# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Encode output labels categorically
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

# CReate SVM model
model = SVC(kernel="linear", probability=True)

model.fit(trainX, trainY)

save_obj("face_recog_model.pickle", model)
save_obj("label_encoder.pickle", out_encoder)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainY, yhat_train)

score_test = accuracy_score(testY, yhat_test)

print("Accuracy: Train={:.3f}, Test={:.3f}".format(score_train*100, score_test*100))