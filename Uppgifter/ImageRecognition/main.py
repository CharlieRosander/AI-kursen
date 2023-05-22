import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.callbacks import EarlyStopping

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
            img = cv2.resize(img, (84, 84))  # resize image to 64x64 pixels
            img = img.astype('float32') / 255  # normalize pixel values
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    return images, labels, filenames


# load images
images, labels, filenames = load_images('./images/train')

# convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)
filenames = np.array(filenames)

# one-hot encoding of labels
labels = to_categorical(labels, 2)

# split into train and test sets
X_train, X_test, y_train, y_test, train_filenames, test_filenames = train_test_split(images, labels, filenames,
                                                                                     test_size=0.2, random_state=42)


# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 3)),  # Input shape: 64x64 pixels, 3 color channels
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Output layer: 2 classes (cats and dogs)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=4)

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[early_stopping])


# Evaluate the model using the test data
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
random_indices = np.random.choice(X_test.shape[0], size=10)

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
accuracy = (correct_guesses / 10) * 100

print(f"{correct_guesses}/10 were guessed correctly, giving an overall accuracy of {accuracy}%")


