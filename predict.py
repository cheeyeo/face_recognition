# Predict identity given image
import numpy as np
import argparse
from model import face_model, triplet_loss, facenet_keras_model
from utils import load_obj, extract_face, get_embedding
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to perform recognition.")
# TODO: Replace below based on width/height of model input layer...
ap.add_argument("-rq", "--required_size", default="96", required=True, help="Size of extracted facial pixels.")
ap.add_argument("-cf", "--channels_first", action="store_true", help="Turn on/off channel first for keras image data format")
ap.add_argument("-m", "--model", type=str, required=True, help="Select model to run.")
ap.add_argument("-s", "--scale", default="normalize", required=True, help="Mode for scaling pixel values. Options are standardize or normalize")
args = vars(ap.parse_args())

# Load the facial detector
detector = MTCNN()

if args["model"] == "facenet":
	print("[INFO] Loading Keras Facenet model...")
	model = facenet_keras_model("model/facenet_keras.h5")
else:
	model = load_model("face_model.h5", custom_objects={"triplet_loss": triplet_loss})

recognizer = load_obj("face_recog_model.pickle")
# print(recognizer.support_vectors_)

le = load_obj("label_encoder.pickle")
print(le.classes_)

size = int(args["required_size"])
face = extract_face(args["image"], detector, required_size=(size, size), align=True)
img = Image.fromarray(face)
img.save('tmp.jpg')
embedding = get_embedding(face, model, args["channels_first"], mode=args["scale"])
embedding = np.expand_dims(embedding, axis=0)

preds = recognizer.predict_proba(embedding)[0]
print(preds)
j = np.argmax(preds)
proba = preds[j]

if proba > 0.7:
	identity = le.classes_[j]
	print("[INFO] Identity: {}, Prob: {:.3f}".format(identity, proba))
else:
	print("[INFO] Unknown identity.")