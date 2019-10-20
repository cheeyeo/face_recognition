# Face Recognition

## 1. Description

( Describes the problem that is being solved, the source of the data, inputs, and outputs. )

The purpose of this project is to develop a custom face recognition system which is capable of identiying the user given a facial image.

The dataset comprises of custom facial images and facial images of celebrities.

The [5 celebrities faces dataset] comprises of images of 5 well known celebrities categorized into training and validation sets. Each image comprises of a single face shot of the celebrity in different orientation and backgrounds. 

The custom dataset comprises of face shots taken using a custom openCV script in `build_face_dataset.py` which takes a single facial image via the webcam. Different facial orientations and backgrounds are used to capture the images. A total of 25 images are used. These are split into training and validation sets similar to the celebrities dataset with at least 20 images for training and 5 for validation, from a 80-20 split.

Before training, each image from both the training and validation sets are pre-processed using the MTCNN model to extract just the faces which are then fed into a pre-trained Facenet model to create the facial embedding of its features i.e. eyes, nose

These embedding features are serialized and used in the training process as input.

The output of the model will be the identiy of each image which corresponds to the sub-directory name the image is stored in. These are one-hot encoded and used as the output labels in the training phase. 

## 2. Test harness

( Describes how model selection will be performed including the resampling method and model evaluation metrics. )

For the baseline model, we will be calculating its classification accuracy based on whether it is able to correctly identity a given image with its corresponding identity.

Other evaluation metrics used will include F1 scores, ROC curves and area-under-curves.

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

* [5 celebrities faces dataset]: https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset