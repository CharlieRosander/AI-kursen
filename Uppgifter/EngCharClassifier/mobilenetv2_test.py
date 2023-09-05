import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

#################### VARS ######################
directory_path = "./EngChars/Img/"
csv_path = "./EngChars/english.csv"
own_csv_path = "./EngChars/oc_testlabels.csv"
own_directory_path = "./EngChars/oc_testdata/"
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

################################################

def label_to_int(label):
    if label.isdigit():
        return int(label)
    elif label.isupper():
        return 10 + ord(label) - ord('A')
    elif label.islower():
        return 36 + ord(label) - ord('a')


def load_image_from_row(row, directory_path):
    img_path = os.path.join(directory_path, row.image[4:])
    label = row.label

    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    img = mobilenet_v2.preprocess_input(img)  # Use MobileNetV2 preprocessing

    label_index = label_to_int(label)

    return img, label_index

def load_images_from_csv(csv_path, directory_path):
    images = []
    labels = []

    df = pd.read_csv(csv_path)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda row: load_image_from_row(row, directory_path), df.itertuples(index=False)))

    for img, label_index in results:
        images.append(img)
        labels.append(label_index)

    images = np.array(images)
    labels = to_categorical(np.array(labels))

    return images, labels

# For loading your own test images

# Specific row-loading function for own tests
def load_own_test_image_from_row(row, directory_path):
    img_path = os.path.join(directory_path, row.image.replace("oc_testdata/", ""))
    label = row.label

    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    img = mobilenet_v2.preprocess_input(img)  # Use MobileNetV2 preprocessing

    label_index = label_to_int(label)

    return img, label_index

def load_own_test_images_from_csv(own_csv_path, own_directory_path):
    images = []
    labels = []

    df = pd.read_csv(own_csv_path)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda row: load_own_test_image_from_row(row, own_directory_path), df.itertuples(index=False)))

    for img, label_index in results:
        images.append(img)
        labels.append(label_index)

    images = np.array(images)
    labels = to_categorical(np.array(labels))

    return images, labels

# def preprocess_mobilenetv2(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (96, 96))
#     img = preprocess_input(img)  # MobileNetV2 specific preprocessing
#     img = np.expand_dims(img, axis=0)
#     return img

# Load images and labels
images, labels = load_images_from_csv(csv_path, directory_path)

# Create training, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Load MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
output_layer = Dense(62, activation='softmax')(x)  # 62 classes

# Create the new model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Evaluate on your own test set
own_test_images, own_test_labels = load_own_test_images_from_csv(own_csv_path, own_directory_path)
print(own_test_images.shape, own_test_labels.shape)
own_test_loss, own_test_acc = model.evaluate(own_test_images, own_test_labels)
print(f"Test accuracy on own test data: {own_test_acc}")

# Save the model
model.save('models/mobilenetv2_model.h5')