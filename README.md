# Face Recognition

## 1. Description

( Describes the problem that is being solved, the source of the data, inputs, and outputs. )

The purpose of this project is to develop a custom face recognition system which is capable of identiying the user given a facial image.

The dataset comprises of custom facial images and facial images of celebrities.

The [5 celebrities faces dataset] comprises of images of 5 well known celebrities categorized into training and validation sets. Each image comprises of a single face shot of the celebrity in different orientation and backgrounds. 

The custom dataset comprises of face shots taken using a custom openCV script in `build_face_dataset.py` which takes a single facial image via the webcam. Different facial orientations and backgrounds are used to capture the images. A total of 25 images are used. These are split into training and validation sets similar to the celebrities dataset with at least 20 images for training and 5 for validation, from a 80-20 split.

Before training, each image from both the training and validation sets are pre-processed using the MTCNN model to extract just the faces which are then fed into a pre-trained [Facenet model] to create the facial embedding of its features i.e. eyes, nose

These embedding features are serialized and used in the training process as input.

The output of the model will be the identiy of each image which corresponds to the sub-directory name the image is stored in. These are one-hot encoded and used as the output labels in the training phase. 

## 2. Test harness

( Describes how model selection will be performed including the resampling method and model evaluation metrics. )

A suitable single value evaluation metric could be the accuracy of the model in identifying the specific individual given a single facial image of that individual. 

For instance, if the dataset comprises of 5 individuals, a high performant model should return `[1,0,0,0,0]` given a facial image of the first individual.

Hence, we are measuring the subset accuracy in predicting that specific individual rather than across the entire dataset.

The `accuracy_score` function is used to evaluate the true/correct labels from the dataset against the model's predictions.

## 3. Baseline performance

( Describes the baseline model performance (using the test harness) that defines whether a model is skillful or not. )

The baseline model is a single linear SVM clasisfier which will take as input the facial embeddings and the one-hot encoded identities as output. 

During training, the goal of the SVM classifier will be to learn enough distinct features from the embeddings in order to be able to separate them linearly since similar embeddings belonging to the same identity should be closer in distance in the geometric space to each other.

After training, the model is evaluated against the unseen validation set.

Accuracy on the training set is `77.876` and accuracy on the validation set is `70.0`


## 4. Experimental Results

( Presents the experimental results, perhaps testing a suite of models, model configurations, data preparation schemes, and more. Each subsection should have some form of:
4.1 Intent: why run the experiment?
4.2 Expectations: what was the expected outcome of the experiment?
4.3 Methods: what data, models, and configurations are used in the experiment?
4.4 Results: what were the actual results of the experiment?
4.5 Findings: what do the results mean, how do they relate to expectations, what other experiments do they inspire? )

### 4.1 Facial Alignment

#### Intent
The extracted facial images are of different orientations and scale. This would affect the quality of the embeddings generated as it would not be able to capture the salient facial features accurately.

Facial alignment attempts to create more accurate facial embeddings by aligning each extracted face in the same position based on the horizontal alignment of the eyes.

In doing so, it is hoped that it would improve overall classifier accuracy.

### Methods

A custom function was developed using `openCV` which applies an affine transformation to the original images which is then processed by the MTCNN network to extract the facial features.

The code is adapted from the example show in this [OpenCV Face Recognition] tutorial.

Examples of applying the transformation can be seen below:

Original Image | Aligned Image
-------------- | -------------
![Original image](artifacts/orig.jpg) | ![Aligned image](artifacts/orig_aligned.jpg)


The aligned images are resized into (96, 96) dimensions which is a requirement input size for the Facenet model. Embeddings are created from these aligned images and serialized to disk using the same methods for data preparation during the training process.

### Expectations / Results

The main expectation is that there should be an improvement on the model's overally accuracy scores. This is because the faces are centered in the image; the eyes are horizontally aligned; and the scaled faces are similar in size.

Running the training script shows that the overall model accuracy has improved to the following:

```
Accuracy: Train=88.496, Test=83.333
```

This means that the model is able to correctly predict 88% of the training data compared to the expected training labels; correctly predict 83% of the validation data compared to the expected validation labels.

( TODO: Use PCA to visualize quality of aligned embeddings compared to non-aligned embeddings. )

## 5. Improvements

( Describes experimental results for attempts to improve the performance of the better performing models, such as hyperparameter tuning and ensemble methods. )

## 6. Final Model

( Describes the choice of a final model, including configuration and performance. It is a good idea to demonstrate saving and loading the model and demonstrate the ability to make predictions on a holdout dataset. )

## 7. Extensions

( Describes areas that were considered but not addressed in the project that could be explored in the future. )

## 8. Resources

* [Deep Learning for Computer Vision]: https://machinelearningmastery.com/deep-learning-for-computer-vision

* [Convolutional Model course from deeplearning.ai]: https://www.coursera.org/learn/convolutional-neural-networks

* [How to collate face recognition dataset]: https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

* [How to detect faces from video stream in open cv]: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

* [OpenCV Face Recognition]: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition

* [5 celebrities faces dataset]: https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset

* [FaceNet model]: https://github.com/iwantooxxoox/Keras-OpenFace/