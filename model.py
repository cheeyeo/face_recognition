import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense

from inception_blocks_v2 import *

def facenet_keras_model(file):
  model = load_model(file)
  return model

def triplet_loss(y_true, y_pred, alpha=0.2):
  """
  Implementation of triplet_loss function

  y_true is required when defining custom loss function in Keras but not used here..
  """
  anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

  # Compute distance between anchor and positive, sum over axis=-1
  pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)

  # Compute distance between anchor and negative, sum over axis=-1
  neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

  # Compute difference and add alpha
  basic_loss = pos_dist - neg_dist + alpha

  # Take max of loss and sum over training examples
  loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

  return loss

def face_model(input_shape):
  """
  Returns a Inception model pretrained
  """

  X_input = Input(shape=input_shape)

  # Zero-padding
  X = ZeroPadding2D((3, 3))(X_input)

  # First block
  X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
  X = BatchNormalization(axis=1, name='bn1')(X)
  X = Activation('relu')(X)

  # Zero-Padding + MAXPOOL
  X = ZeroPadding2D((1, 1))(X)
  X = MaxPooling2D((3, 3), strides=2)(X)

  # Second Block
  X = Conv2D(64, (1, 1), strides=(1, 1), name='conv2')(X)
  X = BatchNormalization(axis=1, epsilon=0.00001, name='bn2')(X)
  X = Activation('relu')(X)

  # Zero-Padding
  X = ZeroPadding2D((1, 1))(X)

  # Third Block
  X = Conv2D(192, (3, 3), strides=(1, 1), name='conv3')(X)
  X = BatchNormalization(axis=1, epsilon=0.00001, name='bn3')(X)
  X = Activation('relu')(X)

  # Zero-Padding + MAXPOOL
  X = ZeroPadding2D((1, 1))(X)
  X = MaxPooling2D(pool_size=3, strides=2)(X)

  # Inception 1: a/b/c
  X = inception_block_1a(X)
  X = inception_block_1b(X)
  X = inception_block_1c(X)

  # Inception 2: a/b
  X = inception_block_2a(X)
  X = inception_block_2b(X)

  # Inception 3: a/b
  X = inception_block_3a(X)
  X = inception_block_3b(X)

  # Top layer
  X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
  X = Flatten()(X)
  X = Dense(128, name='dense_layer')(X)

  # L2 normalization
  X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

  model = Model(inputs=X_input, outputs=X, name='FaceRecogModel')

  model.compile(loss=triplet_loss, optimizer='adam', metrics=['accuracy'])

  return model


if __name__ == "__main__":
  with tf.Session() as test:
    tf.compat.v1.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random.normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random.normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)
    
    print("Test Triplet Loss = " + str(loss.eval()))


  m = face_model((3, 96, 96))
  m.summary()
  print(m.count_params())