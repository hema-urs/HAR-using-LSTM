# HAR-using-LSTM
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
!wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
!unzip -q "UCI HAR Dataset.zip"

def load_signals(folder, split='train'):
    signals = []
    signal_files = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    for signal in signal_files:
        path = f"{folder}/Inertial Signals/{signal}_{split}.txt"
        data = np.loadtxt(path)
        signals.append(data)
    return np.transpose(np.array(signals), (1, 2, 0))
X_train = load_signals("UCI HAR Dataset/train", split='train')
X_test = load_signals("UCI HAR Dataset/test", split='test')
y_train = np.loadtxt("UCI HAR Dataset/train/y_train.txt")
y_test = np.loadtxt("UCI HAR Dataset/test/y_test.txt")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

y_train_cat = tf.keras.utils.to_categorical(y_train_encoded)
y_test_cat = tf.keras.utils.to_categorical(y_test_encoded)
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, epochs=15, batch_size=64, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

print(classification_report(y_test_encoded, y_pred_labels, target_names=[
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]))
