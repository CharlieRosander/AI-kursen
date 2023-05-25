import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Variables to hold the number of train and test images and epochs.
epochs = 15
train_img_num = 3000
test_img_num = int(train_img_num * 0.3)
batch_size = 32
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_images(folder, limit):
    images = []
    labels = []
    filenames = []
    cats_counter = 0
    dogs_counter = 0
    for filename in os.listdir(folder):
        if filename.startswith('cat') and cats_counter < limit / 2:
            label = 0
            cats_counter += 1
        elif filename.startswith('dog') and dogs_counter < limit / 2:
            label = 1
            dogs_counter += 1
        else:
            continue
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (84, 84))
            img = img.astype('float32') / 255
            images.append(img)
            labels.append(label)
            filenames.append(filename)
    return images, labels, filenames


X_train, y_train, train_filenames = load_images('./images/train', train_img_num)
print(f"Loaded {train_img_num} train images")
X_test, y_test, test_filenames = load_images('./images/test', test_img_num)
print(f"Loaded {test_img_num} test images")

X_train = np.array(X_train)
y_train = to_categorical(np.array(y_train), 2)

X_test = np.array(X_test)
y_test = to_categorical(np.array(y_test), 2)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def get_previous_accuracy(file_name):
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            last_accuracy_line = [line for line in lines if "Test accuracy" in line]
            if not last_accuracy_line:  # If no accuracy line is found, it's the first run
                return 0
            else:  # Otherwise, return the last accuracy
                last_accuracy = float(last_accuracy_line[-1].split(":")[1].strip())
                return last_accuracy
    except FileNotFoundError:
        return 0  # If file does not exist, it's the first run


early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
history = tf.keras.callbacks.History()
saved_model_path = './Models/CatDog_classifier_{timestamp}.h5'
model_is_saved = 0
previous_accuracy = get_previous_accuracy('./Docs/test_results.txt')
latest_accuracy = None
reg_factor = 0.001
reg_method = "l2"
model = None


def create_new_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(84, 84, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, history]
    )

    model.save(saved_model_path)
    print("Model saved.")
    return model


def continue_training_model(saved_model_path):
    model = load_model(saved_model_path)
    print("Continuing training...")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, history]
    )

    # Get the latest accuracy
    latest_accuracy = history.history['val_accuracy'][-1]
    print("The model will only be saved if the accuracy is higher than the previous accuracy.")
    print(f"Previous accuracy: {previous_accuracy}")
    print(f"Latest accuracy: {latest_accuracy}")

    if latest_accuracy > previous_accuracy:
        model.save(saved_model_path)
        print("Model saved.")
    return model


if os.path.isfile(saved_model_path):
    print("A saved model was found.")
    user_input = input("Do you want to continue training this model? (y/n): ")
    if user_input.lower() == 'y':
        model = continue_training_model(saved_model_path)
    elif user_input.lower() == 'n':
        model = create_new_model()
else:
    print("No saved model was found.")
    model = create_new_model()

# Evaluate the model using the test data.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

# Keep track of correct guesses
correct_guesses = 0

# Choose x random test images where x == test_img_num
random_indices = np.random.choice(X_test.shape[0], size=test_img_num)

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
print('\nTest accuracy:', test_acc)
print(f"{correct_guesses}/{test_img_num} were guessed correctly, giving an overall accuracy of {overall_accuracy}%")


# Write the results to a file
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

# Number of epoch at which training was stopped
stopping_epoch = early_stopping.stopped_epoch

# Retrieve training and validation loss and accuracy from the history object
num_epochs = len(history.history['loss'])

# Write the results to a file
with open('./Docs/test_results.txt', 'a') as f:
    # Run Information
    f.write('--- Run Information ---\n')
    f.write(f'Test-run: {run_number}\n')
    f.write('\n')

    # Model Information
    f.write('--- Model Information ---\n')
    if model_is_saved == 1:
        f.write("Model: Previously saved model was used.\n")
        f.write(f'Previous accuracy: {previous_accuracy}\n')
        f.write(f'Latest accuracy: {latest_accuracy}\n')
    else:
        f.write("Model: New model was trained.\n")
    f.write(f'Regularization: {reg_method} with factor {reg_factor}\n')
    f.write('\n')

    # Data Information
    f.write('--- Data Information ---\n')
    f.write(f'Train images: {train_img_num}\n')
    f.write(f'Test images: {test_img_num}\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Epochs run: {num_epochs}\n')
    f.write(f'Early stopping patience: {early_stopping.patience}\n')
    f.write('\n')

    # Results
    f.write('--- Results ---\n')
    f.write(f'Correct guesses: {correct_guesses}/{test_img_num}\n')
    f.write(f'Test accuracy: {test_acc}\n')
    f.write(f'Overall accuracy: {overall_accuracy:.3f}%\n')
    if stopping_epoch > 0:
        f.write(f'Training stopped early at epoch: {stopping_epoch}\n')
    else:
        f.write("Training completed and did not stop early.\n")
    f.write('\n\n')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
