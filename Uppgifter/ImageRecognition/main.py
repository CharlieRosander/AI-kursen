import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l1, l2
from keras.preprocessing.image import ImageDataGenerator

# Variables to hold the number of train and test images and epochs.
epochs = 30
train_img_num = 2000  # Adjust this variable to change the number of training images
test_img_num = int(train_img_num * 0.3)  # Adjust this variable to change the number of test images
batch_size = 32


# define function to load the images
def load_images(folder, limit):
    images = []
    labels = []
    filenames = []
    cats_counter = 0
    dogs_counter = 0
    for filename in os.listdir(folder):
        if filename.startswith('cat') and cats_counter < limit / 2:  # Load 'cat' images
            label = 0
            cats_counter += 1
        elif filename.startswith('dog') and dogs_counter < limit / 2:  # Load 'dog' images
            label = 1
            dogs_counter += 1
        else:
            continue

        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (84, 84))  # resize image to 84x84 pixels
            img = img.astype('float32') / 255  # normalize pixel values
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    return images, labels, filenames


# Load the train and test images.
X_train, y_train, train_filenames = load_images('./images/train', train_img_num)
print(f"Loaded {train_img_num} train images")
X_test, y_test, test_filenames = load_images('./images/test', test_img_num)
print(f"Loaded {test_img_num} test images")

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
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Create an instance of the History callback
history = tf.keras.callbacks.History()

# Train the model.
model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) / 32,
    epochs=epochs,  # Increase the number of epochs
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, history]
)

# Number of epoch at which training was stopped
stopping_epoch = early_stopping.stopped_epoch

# Then you can retrieve the number of epochs like this:
num_epochs = len(history.history['loss'])

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
random_indices = np.random.choice(X_test.shape[0], size=test_img_num)

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
overall_accuracy = (correct_guesses / test_img_num) * 100

print(f"{correct_guesses}/{test_img_num} were guessed correctly, giving an overall accuracy of {overall_accuracy}%")


def get_next_run_number(file_name):
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            last_run_line = [line for line in lines if "Test-run" in line]
            if not last_run_line:  # If no Test-run line is found, it's the first run
                return 1
            else:  # Otherwise, increment the last run number by one
                last_run_number = int(last_run_line[-1].split(":")[1].strip())
                return last_run_number + 1
    except FileNotFoundError:
        return 1  # If file does not exist, it's the first run


# Get the run number
run_number = get_next_run_number('./Docs/test_results.txt')

# Write the results to a file
with open('./Docs/test_results.txt', 'a') as f:
    f.write(f'Test-run: {run_number}\n')
    f.write(f'Train images: {train_img_num}\n')
    f.write(f'Test images: {test_img_num}\n')
    f.write(f'Epochs run: {num_epochs}\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Early stopping patience: {early_stopping.patience}\n')
    f.write(f'Correct guesses: {correct_guesses}/{test_img_num}\n')
    f.write(f'Test accuracy: {test_acc}\n')
    f.write(f'Overall accuracy: {overall_accuracy:.3f}%\n')
    if stopping_epoch > 0:
        f.write(f'Training stopped early at epoch: {stopping_epoch}\n')
    else:
        f.write("Training completed and did not stop early.\n")
    f.write('\n\n')
