import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2, l1_l2
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import pandas as pd

#################### VARS ######################
regularization_rate = 0.001
epoch_num = 35
batch_size = 16
directory_path = "./EngChars/Img/"
csv_path = "./EngChars/english.csv"
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
    img = cv2.resize(img, (84, 84))
    img = img / 255.0  # normalize

    label_index = label_to_int(label)

    return img, label_index


def load_images_from_csv(csv_path, directory_path):
    images = []
    labels = []

    df = pd.read_csv(csv_path)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda row: load_image_from_row(
            row, directory_path), df.itertuples(index=False)))

    for img, label_index in results:
        images.append(img)
        labels.append(label_index)

    images = np.array(images)
    labels = to_categorical(np.array(labels))

    return images, labels


# Load the images and labels
images, labels = load_images_from_csv(csv_path, directory_path)

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Kollar så att det ser ut som vi förväntar oss, vilket det gör

print(x_train.shape)
print(x_test.shape)


# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Dropout added

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Dropout added

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))  # Dropout added

model.add(Flatten())
model.add(Dense(128, activation='relu',
          kernel_regularizer=l2(regularization_rate)))
model.add(Dropout(0.5))  # Dropout added

model.add(Dense(62, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=epoch_num,
                    batch_size=batch_size, validation_data=(x_test, y_test))

# Save model, create dir if not exists
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/model.h5')

# Plot accuracy and loss
if not os.path.exists('./Plots/'):
    os.makedirs('./Plots/')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("./Plots/acc_vals.jpg")

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("./Plots/loss_vals.jpg")

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))


# Plot random image sample with predicted label
def int_to_label(i):
    if i < 10:
        return str(i)
    elif i < 36:
        return chr(i - 10 + ord('A'))
    else:
        return chr(i - 36 + ord('a'))


# Get predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Convert categorical array to original label
actual_labels = np.argmax(y_test, axis=1)

# Choose random indices to visualize
random_indices = np.random.choice(x_test.shape[0], size=9, replace=False)

plt.figure(figsize=(10, 10))

for i, index in enumerate(random_indices):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[index])
    plt.title(
        f"Actual: {int_to_label(actual_labels[index])}\nPredicted: {int_to_label(predicted_labels[index])}")
    plt.axis('off')

plt.tight_layout()
plt.show()
