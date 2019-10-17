import numpy as np
from utils import load_dataset

print('[INFO] Loading data...')
directory = 'data'

trainX, trainY = load_dataset('data/train')
testX, testY = load_dataset('data/val')

np.savez_compressed('faces_dataset.npz', trainX, trainY, testX, testY)

data = np.load('faces_dataset.npz')
trainX, trainY = data['arr_0'], data['arr_1']
testX, testY = data['arr_2'], data['arr_3']

print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)