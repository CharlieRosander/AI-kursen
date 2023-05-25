import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1, l2, l1_l2
from datetime import datetime


class DataHandler:
    def __init__(self, train_img_num, test_img_num, batch_size):
        self.train_img_num = train_img_num
        self.test_img_num = test_img_num
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_filenames = None
        self.test_filenames = None
        self.history = tf.keras.callbacks.History()
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def load_images(self, folder, limit):
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

    def load_training_and_test_data(self):
        train_folder = './images/train'
        test_folder = './images/test'

        # Check if directories exist, if not, create them
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        self.X_train, self.y_train, self.train_filenames = self.load_images(train_folder, self.train_img_num)
        print(f"Loaded {self.train_img_num} train images")
        self.X_test, self.y_test, self.test_filenames = self.load_images(test_folder, self.test_img_num)
        print(f"Loaded {self.test_img_num} test images")


class ModelHandler:
    def __init__(self, epochs, data_handler, timestamp, regularization_method=None, regularization_factor=None):
        self.regularization_method = regularization_method
        self.regularization_factor = regularization_factor
        self.epochs = epochs
        self.data_handler = data_handler
        self.saved_model_path = f'./Models/CatDog_classifier_{timestamp}.h5'
        self.model = None
        self.history = None
        self.early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)


    def create_new_model(self):
        models_folder = './Models'

        # Check if directories exist, if not, create them
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        regularizer = None
        if self.regularization_method is not None and self.regularization_factor is not None:
            if self.regularization_method.lower() == 'l1':
                regularizer = l1(self.regularization_factor)
            elif self.regularization_method.lower() == 'l2':
                regularizer = l2(self.regularization_factor)
            elif self.regularization_method.lower() == 'l1_l2':
                regularizer = l1_l2(self.regularization_factor)

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(84, 84, 3),
                   kernel_regularizer=regularizer),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=regularizer),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.train_new_model()

    def train_new_model(self):
        self.history = self.model.fit(
            self.data_handler.datagen.flow(self.data_handler.X_train, self.data_handler.y_train,
                                           batch_size=self.data_handler.batch_size),
            steps_per_epoch=len(self.data_handler.X_train) / self.data_handler.batch_size,
            epochs=self.epochs,
            validation_data=(self.data_handler.X_test, self.data_handler.y_test),
            callbacks=[self.early_stopping]
        )

        input("Do you want to save the model? y/n: ")
        if input() == 'y':
            self.model.save(self.saved_model_path)
            print("Model saved.")
        else:
            print("Model not saved. Continuing...")

    def train_existing_model(saved_model_path):
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
            with open('./Docs/test_results.txt', 'a') as f:
                f.write(f"Latest accuracy: {latest_accuracy}\n")
        return model

    def handle_model_training(self):
        if os.path.isfile(self.saved_model_path):
            print("A saved model was found.")
            user_input = input("Do you want to continue training this model? (y/n): ")
            if user_input.lower() == 'y':
                self.train_existing_model()
            elif user_input.lower() == 'n':
                self.create_new_model()
        else:
            print("No saved model was found. Creating and training a new model.")
            self.create_new_model()


class Tester:
    def __init__(self, data_handler, model_handler, timestamp):
        self.data_handler = data_handler
        self.model_handler = model_handler
        self.timestamp = timestamp
        self.correct_guesses = 0

    def test_model(self):
        # Evaluate the model using the test data.
        test_loss, test_acc = self.model_handler.model.evaluate(self.data_handler.X_test, self.data_handler.y_test,
                                                                verbose=2)
        # Choose x random test images where x == test_img_num
        random_indices = np.random.choice(self.data_handler.X_test.shape[0], size=self.data_handler.test_img_num)

        for i in random_indices:
            test_image = self.data_handler.X_test[i]
            test_label = self.data_handler.y_test[i]

            # Add an extra dimension to the image tensor
            test_image = np.expand_dims(test_image, axis=0)

            # Use the model to predict the label of the test image
            prediction = self.model_handler.model.predict(test_image)

            # The prediction is an array of probabilities for each class.
            # The class with the highest probability is the model's prediction
            predicted_label = np.argmax(prediction)

            if predicted_label == np.argmax(test_label):
                self.correct_guesses += 1

        # Calculate accuracy
        overall_accuracy = (self.correct_guesses / self.data_handler.test_img_num) * 100
        print('\nTest accuracy:', test_acc)
        print(f"{self.correct_guesses}/{self.data_handler.test_img_num} were guessed correctly, giving an overall "
              f"accuracy of {overall_accuracy}%")

        self.plot_training_results()

    def write_log(run_number, model_is_saved, previous_accuracy, latest_accuracy, reg_method, reg_factor,
                  train_img_num, test_img_num, batch_size, num_epochs, early_stopping, correct_guesses,
                  test_acc, overall_accuracy, stopping_epoch):
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

    def plot_training_results(self):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.model_handler.history.history['accuracy'])
        plt.plot(self.model_handler.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.model_handler.history.history['loss'])
        plt.plot(self.model_handler.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()


# Ask user about regularization
use_regularization = input("Do you want to use regularization? (y/n): ")
if use_regularization.lower() == 'y':
    regularization_method = input("Please enter regularization method (l1, l2, l1_l2): ")
    regularization_factor = float(input("Please enter regularization factor (a positive float number): "))
else:
    regularization_method = None
    regularization_factor = None

#############################################
# Define constants and hyperparameters
train_img_num = 3000
test_img_num = int(train_img_num * 0.2)
batch_size = 32
epochs_amount = 15
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#############################################

data_handler = DataHandler(train_img_num, test_img_num, batch_size)
model_handler = ModelHandler(epochs_amount, data_handler, timestamp, regularization_method, regularization_factor)
previous_accuracy = model_handler.get_previous_accuracy("./Docs/test_results.txt")
tester = Tester(data_handler, model_handler, timestamp)

# Load Training and Testing Data
data_handler.load_training_and_test_data()

# Train Model
model_handler.handle_model_training()

# Test Model
tester.test_model()
