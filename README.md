# Bharatanatyam Mudra Recognition - CNN Classification

## Project Overview
This project is a 5-class image classification task for Bharatanatyam mudras. The model is trained to recognize the five mudra classes: Musti, Pataka, Sikhara, Simhamukha, and Trisula.

## Dataset
The notebook loads images from a folder structure containing the five mudra classes. After cleaning the dataset, the final training data contains 2000 images in total, with 400 images per class. The class distribution from the notebook is:

- Musti: 400
- Pataka: 400
- Sikhara: 400
- Simhamukha: 400
- Trisula: 400

The notebook also checks the dataset and notes that the source folder contains extra folders such as `train` and `test`, but the final dataset is filtered to the five mudra classes used for model training.

## Workflow
The workflow in the notebook is:

- dataset analysis and class checking
- preprocessing the image data
- splitting into train, validation, and test sets
- training a custom CNN model
- evaluating the model on the test set
- checking confusion and misclassified samples

## Preprocessing and Augmentation
The image preprocessing in the notebook uses `ImageDataGenerator(rescale=1./255)`. The images are resized to 128x128 and normalized to the [0, 1] range before training. The notebook content reviewed here does not include additional augmentation steps beyond this rescaling step.

## CNN Model
The model is a custom Keras CNN built for 5-class classification. It takes 128x128 RGB images as input and ends with a softmax output layer for the five mudra classes. The notebook uses a standard CNN structure with convolutional layers, pooling, and dense layers, and the training is run with TensorFlow/Keras.

## Results
The notebook evaluates the model using accuracy, precision, recall, F1-score, and confusion matrix on the held-out test set. The exact metric values are produced in the notebook output during evaluation, and the model is checked with classification reports and confusion analysis.

## Error Analysis
The notebook checks the class distribution and then reviews the model using confusion and misclassification analysis. The final class distribution is balanced across the five mudra classes, and the confusion matrix is used to inspect which classes are more likely to be confused with each other.

## Additional Experiment
The notebook imports transfer-learning models such as `MobileNetV2`, `EfficientNetB0`, and `ResNet50`, but no completed transfer-learning experiment or final comparison results are present in the notebook output included in this repository.

## Deployment
The repository contains a Streamlit app in `app.py` for image upload and mudra prediction. It is designed to load the model and class labels and show the predicted class along with the confidence score.

## Repository Structure

```text
Exit_Exam/
├── MUDRA_CNN.ipynb
├── README.md
├── app.py
├── class_indices.json
├── model_metadata.json
```

## How to Run
From the project folder, run:

```bash
streamlit run app.py
```

This app expects the model files and label information to be available in the folder where it is run.
