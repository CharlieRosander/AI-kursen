import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator


# define function to load the images
def load_images(folder):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.startswith('cat'):
            label = 0
        else:
            label = 1
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (84, 84))  # resize image to 84x84 pixels
            img = img.astype('float32') / 255  # normalize pixel values
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    return images, labels, filenames


# Load the train and test images.
X_train, y_train, train_filenames = load_images('./images/train')
X_test, y_test, test_filenames = load_images('./images/test')

# Convert lists to numpy arrays, and apply one-hot encoding to the labels.
X_train = np.array(X_train)
y_train = to_categorical(np.array(y_train), 2)

X_test = np.array(X_test)
y_test = to_categorical(np.array(y_test), 2)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(84, 84, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Early stopping with a higher patience.
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model.
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    epochs=20,  # Increase the number of epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the model using the test data.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# Choose a random test image
random_index = np.random.choice(X_test.shape[0])

test_image = X_test[random_index]
test_label = y_test[random_index]
test_filename = test_filenames[random_index]

# Add an extra dimension to the image tensor
test_image = np.expand_dims(test_image, axis=0)

# Use the model to predict the label of the test image
prediction = model.predict(test_image)

# The prediction is an array of probabilities for each class.
# The class with the highest probability is the model's prediction
predicted_label = np.argmax(prediction)

# Choose 10 random test images
# Choose 10 random test images
random_indices = np.random.choice(X_test.shape[0], size=100)

# Keep track of correct guesses
correct_guesses = 0

# Print the tested image's filename, predicted and true labels
for i in random_indices:
    test_image = X_test[i]
    test_label = y_test[i]
    test_filename = test_filenames[i]

    # Add an extra dimension to the image tensor
    test_image = np.expand_dims(test_image, axis=0)

    # Use the model to predict the label of the test image
    prediction = model.predict(test_image)

    # The prediction is an array of probabilities for each class.
    # The class with the highest probability is the model's prediction
    predicted_label = np.argmax(prediction)

    if predicted_label == np.argmax(test_label):
        correct_guesses += 1

# Calculate accuracy
accuracy = (correct_guesses / 100) * 100

print(f"{correct_guesses}/100 were guessed correctly, giving an overall accuracy of {accuracy}%")


