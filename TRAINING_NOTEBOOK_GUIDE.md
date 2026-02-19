# 🧘 Pilates Posture Recognition - Training Notebook

### What This Notebook Does:
1. **Collects** training data from videos/images of Pilates poses
2. **Extracts** body landmarks using MediaPipe
3. **Calculates** joint angles from those landmarks
4. **Trains** a machine learning model to recognize postures
5. **Exports** the model for your Android app

### No ML Experience Needed!
Just run each cell in order. 😊

---  

## CELL 1 (TEXT/MARKDOWN)
### Introduction
This notebook is designed to help you recognize Pilates postures using machine learning. Follow the instructions in each cell for a smooth experience.

---  

## CELL 2 (CODE)
```python
!pip install mediapipe opencv-python tensorflow scikit-learn matplotlib numpy pandas -q
print("✅ All libraries installed successfully!")
```
### Libraries Installation
Ensure that all necessary libraries are installed. Execute the cell above to get started. 

---  

## CELL 3 (CODE)
```python
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("✅ Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"MediaPipe version: {mp.__version__}")
```
### Importing Libraries
This cell imports all necessary libraries for the project. Make sure it runs successfully.

---  

## CELL 4 (CODE)
```python
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe Pose initialized!")
```
### MediaPipe Initialization
In this cell, we initialize MediaPipe Pose. Ensure that the initialization is successful before proceeding to the next steps.

---  

## CELL 5 (CODE)
```python
def calculate_angle(point_a, point_b, point_c):
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

test_angle = calculate_angle([0, 0], [1, 0], [1, 1])
print(f"✅ Angle calculation works! Test: {test_angle:.1f}°")
```
### Angle Calculation Function
This function calculates the angle between three points. Test this functionality to ensure accuracy.

---  

## CELL 6 (CODE)
```python
def extract_pose_features(landmarks):
    try:
        ls = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        le = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        lw = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        lh = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        lk = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        la = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        rs = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        re = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rw = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        rh = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        rk = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ra = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angles = [calculate_angle(ls, le, lw), calculate_angle(rs, re, rw), calculate_angle(le, ls, lh), calculate_angle(re, rs, rh), calculate_angle(ls, lh, lk), calculate_angle(rs, rh, rk), calculate_angle(lh, lk, la), calculate_angle(rh, rk, ra), calculate_angle(lh, ls, rs), calculate_angle(rh, rs, ls)]
        return angles
    except Exception as e:
        return [0] * 10

print("✅ Feature extraction function created!")
```
### Pose Feature Extraction
This function extracts features from pose landmarks, returning angles relevant for posture recognition.

---  

## CELL 7 (CODE)
```python
DATASET_PATH = '/content/pilates_dataset'
print(f"Dataset path: {DATASET_PATH}")
```
### Dataset Path
Set the path for your dataset. Ensure it points to the correct location.

---  

## CELL 8 (CODE)
```python
def process_dataset(path):
    X, y = [], []
    for pose_name in os.listdir(path):
        pose_path = os.path.join(path, pose_name)
        if os.path.isdir(pose_path):
            for file in os.listdir(pose_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img = cv2.imread(os.path.join(pose_path, file))
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        result = pose.process(rgb)
                        if result.pose_landmarks:
                            X.append(extract_pose_features(result.pose_landmarks.landmark))
                            y.append(pose_name)
    return np.array(X), np.array(y)

X, y = process_dataset(DATASET_PATH)
print(f'Collected {len(X)} samples')
```
### Processing Dataset
This function processes the dataset, extracting pose features and storing them along with their respective labels.

---  

## CELL 9 (CODE)
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
```
### Encoding Labels
This cell encodes the labels into numerical format suitable for training. It also splits the data into training and testing sets.

---  

## CELL 10 (CODE)
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f'Training accuracy: {model.score(X_train, y_train):.2%}')
print(f'Testing accuracy: {model.score(X_test, y_test):.2%}')
```
### Random Forest Model Training
This cell trains a Random Forest model and prints the accuracy metrics for both the training and testing datasets.

---  

## CELL 11 (CODE)
```python
nn_model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)), tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(len(le.classes_), activation='softmax')])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)
```
### Neural Network Training
This cell builds and trains a neural network to recognize Pilates postures, outputting its performance during training.

---  

## CELL 12 (CODE)
```python
converter = tf.lite.TFLiteConverter.from_keras_model(nn_model)
tflite_model = converter.convert()
with open('pilates_model.tflite', 'wb') as f:
    f.write(tflite_model)
print('✅ Model saved')
```
### Model Conversion
This cell converts the trained model into the TensorFlow Lite format for deployment on mobile devices. Confirm that it saves without errors.

---  

## CELL 13 (CODE)
```python
with open('labels.txt', 'w') as f:
    for label in le.classes_:
        f.write(label + '\n')
print('✅ Labels saved')
```
### Saving Labels
This cell saves the class labels to a text file, which will be useful for later reference or deployment.

---  

## CELL 14 (CODE)
```python
from google.colab import files
files.download('pilates_model.tflite')
files.download('labels.txt')
print('✅ Files downloaded!')
```
### Download Files
Lastly, this cell allows you to download the saved model and labels. Make sure all files are correctly downloaded to your local system.

---  

### Conclusion
You have now successfully set up a complete training notebook for Pilates posture recognition! Feel free to customize any part of the notebook and explore further!