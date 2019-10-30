import numpy as np
from utils import load_dataset
from PIL import Image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--align", action="store_true", help="Turn on/off alignment of images.")
ap.add_argument("-s", "--size", default="96", help="Set size of face images.")
args = vars(ap.parse_args())

print(args)

print('[INFO] Loading data...')
directory = 'data'

trainX, trainY = load_dataset('data/train', args["size"], align=args["align"])
testX, testY = load_dataset('data/val', args["size"], align=args["align"])

np.savez_compressed('faces_dataset.npz', trainX, trainY, testX, testY)

data = np.load('faces_dataset.npz')
trainX, trainY = data['arr_0'], data['arr_1']
testX, testY = data['arr_2'], data['arr_3']

print("[INFO] trainX shape: {}".format(trainX.shape))
print("[INFO] trainY shape: {}".format(trainY.shape))
print("[INFO] testX shape: {}".format(testX.shape))
print("[INFO] testY shape: {}".format(testY.shape))

# Save orig image for debug
# img = Image.fromarray(trainX[2])
# img.save('artifacts/orig.jpg')

# trainX, _ = load_dataset('data/train', align=True)
# img = Image.fromarray(trainX[2])
# img.save('artifacts/orig_aligned.jpg')