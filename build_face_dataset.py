# Sample script to build face images using webcam
# Reference: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

from imutils.video import VideoStream
import imutils
import os
import time
import argparse
import numpy as np
import cv2
from PIL import Image
import sys

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='Path to output directory.')
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe deploy prototxt file")
ap.add_argument("-m", "--model", required=True, help="Path to pretrained Caffe model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum prob to filter weak detection")
args = vars(ap.parse_args())

# Load Caffe model
# We are using it to detect presence of faces; embeddings are created using MTCNN in create_embedding.py
print("[INFO] Loading cv2 caffe model...")
detector = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize camera stream
print('[INFO] Initializing video camera...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

while True:
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	detector.setInput(blob)
	detections = detector.forward()

	# loop over detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence < args["confidence"]:
			continue

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Draw bounding box of face with prob
		text = "{:.2f}%".format(confidence*100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('k'):
		# Check output dir exists; if not create it
		if not os.path.exists(args['output']):
			os.makedirs(args['output'])

		p = os.path.sep.join([args['output'], '{}.png'.format(str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1
	elif key == ord('q'):
		break

print("[INFO] {} images stored.".format(total))
print("[INFO] Cleaning up...")

cv2.destroyAllWindows()
vs.stop()