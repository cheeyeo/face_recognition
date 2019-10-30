import numpy as np
from utils import save_obj
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

data = np.load("faces_embeddings.npz")
trainX, trainY, testX, testY = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]

print("[INFO] Loaded embeddings shape: trainX: {}, trainY: {}, testX: {}, testY: {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Encode output labels categorically
out_encoder = LabelEncoder()
out_encoder.fit(trainY)
trainY = out_encoder.transform(trainY)
testY = out_encoder.transform(testY)

# Create SVM model
model = SVC(C=1.0, kernel="linear", probability=True, verbose=True)

model.fit(trainX, trainY)
print(model.support_vectors_)

save_obj("face_recog_model.pickle", model)
save_obj("label_encoder.pickle", out_encoder)

yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

score_train = accuracy_score(trainY, yhat_train)

score_test = accuracy_score(testY, yhat_test)
print()
print("Accuracy: Train={:.3f}, Test={:.3f}".format(score_train*100, score_test*100))