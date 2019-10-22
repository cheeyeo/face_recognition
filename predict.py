# Predict identity given image
import numpy as np
import argparse
from model import face_model, triplet_loss
from utils import load_obj, extract_face, get_embedding
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image to perform recognition.")

args = vars(ap.parse_args())

# Load the facial detector
detector = MTCNN()

model = load_model("face_model.h5", custom_objects={"triplet_loss": triplet_loss})

recognizer = load_obj("face_recog_model.pickle")
le = load_obj("label_encoder.pickle")

# print(recognizer.support_vectors_)

face = extract_face(args["image"], detector, align=True)
embedding = get_embedding(face, model)
print(embedding.shape)

embedding = np.expand_dims(embedding, axis=0)
preds = recognizer.predict_proba(embedding)[0]
print(preds)
j = np.argmax(preds)
proba = preds[j]
print(le.classes_)
identity = le.classes_[j]
print("[INFO] Identity: {}, Prob: {:.3f}".format(identity, proba))