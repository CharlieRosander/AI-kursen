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
from typing import Tuple, List


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
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    @staticmethod
    def load_images(folder: str, limit: int) -> Tuple[List[np.array], List[int], List[str]]:
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
        train_folder = './train'
        test_folder = './test'

        # Check if directories exist, if not, create them
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        self.X_train, self.y_train, self.train_filenames = self.load_images(train_folder, self.train_img_num)
        print(f"Loaded {len(self.X_train)} train images")
        self.X_test, self.y_test, self.test_filenames = self.load_images(test_folder, self.test_img_num)
        print(f"Loaded {len(self.X_test)} test images")

        self.X_train = np.array(self.X_train)
        self.y_train = to_categorical(np.array(self.y_train), 2)
        self.X_test = np.array(self.X_test)
        self.y_test = to_categorical(np.array(self.y_test), 2)


class ModelHandler:
    def __init__(self, epochs, data_handler, timestamp, regularization_method=None, regularization_factor=None, patiance=None):
        self.regularization_method = regularization_method
        self.regularization_factor = regularization_factor
        self.epochs = epochs
        self.data_handler = data_handler
        self.saved_model_path = f'./Models/CatDog_classifier.h5'
        self.model = None
        self.history = tf.keras.callbacks.History()
        self.early_stopping = EarlyStopping(monitor='val_accuracy', patience=patiance_value)

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
            Dense(2, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.train_model()
        self.model.save(self.saved_model_path)

    def train_model(self):
        self.history = self.model.fit(
            self.data_handler.datagen.flow(self.data_handler.X_train, self.data_handler.y_train,
                                           batch_size=self.data_handler.batch_size),
            steps_per_epoch=len(self.data_handler.X_train) / self.data_handler.batch_size,
            epochs=self.epochs,
            validation_data=(self.data_handler.X_test, self.data_handler.y_test),
            callbacks=[self.early_stopping]
        )

    def continue_training_model(self):
      models_folder = './Models'

      # Check if directories exist, if not, create them
      if not os.path.exists(models_folder):
          os.makedirs(models_folder)

      previous_accuracy = Tester.get_previous_accuracy()

      if os.path.isfile(self.saved_model_path):
          self.model = load_model(self.saved_model_path)
          print("Continuing training...")
          self.train_model()

          latest_accuracy = self.history.history['val_accuracy'][-1]

          print("The model will only be saved if the accuracy is higher than the previous accuracy.")
          print(f"Previous accuracy: {previous_accuracy}")
          print(f"Latest accuracy: {latest_accuracy}")

          if latest_accuracy > previous_accuracy:
              self.model.save(self.saved_model_path)
              print("\nModel saved.")
          else:
              print("\nModel not saved.")
      else:
          print("No saved model was found, creating a new model.")
          self.create_new_model()


    def handle_model_training(self):
        if os.path.isfile(self.saved_model_path):
            print("\nA saved model was found.")
            user_input = input("Do you want to continue training this model? (y/n): ")
            if user_input.lower() == 'y':
                self.continue_training_model()
            elif user_input.lower() == 'n':
                self.create_new_model()
        else:
            print("No saved model was found.")
            self.create_new_model()


class Tester:
    def __init__(self, data_handler, model_handler, timestamp):
        self.data_handler = data_handler
        self.model_handler = model_handler
        self.timestamp = timestamp
        self.previous_accuracy = self.get_previous_accuracy()

    @staticmethod
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

    @staticmethod
    def get_previous_accuracy():
        try:
            with open('./Docs/test_results.txt', 'r') as f:
                lines = f.readlines()
                last_accuracy_line = [line for line in lines if "Test accuracy" in line]
                if not last_accuracy_line:  # If no accuracy line is found, it's the first run
                    return 0
                else:  # Otherwise, return the last accuracy
                    last_accuracy = float(last_accuracy_line[-1].split(":")[1].strip())
                    return last_accuracy
        except FileNotFoundError:
            return 0

    @staticmethod
    def update_latest_accuracy(test_acc):
        with open('latest_accuracy.txt', 'w') as file:
            file.write(str(test_acc))

    def test_model(self):
        predictions = self.model_handler.model.predict(self.data_handler.X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.data_handler.y_test, axis=1)
        correct_guesses = np.sum(predicted_labels == true_labels)

        test_acc = correct_guesses / self.data_handler.test_img_num
        overall_accuracy = (correct_guesses / self.data_handler.test_img_num) * 100

        acc_improvement = test_acc - self.previous_accuracy

        print(f'Test Accuracy: {test_acc:.3f}')
        print(f'Overall Accuracy: {overall_accuracy:.3f}')
        print(f"Improvement in accuracy compared to previous run: {acc_improvement:.3f}")

        # Update latest accuracy
        self.update_latest_accuracy(test_acc)

        # Record test results
        self.record_test_results(test_acc, correct_guesses, overall_accuracy)

        self.plot_training_results()

    def record_test_results(self, test_acc, correct_guesses, overall_accuracy):
        # Determine the run number
        current_run_number = self.get_next_run_number('./Docs/test_results.txt')

        if not os.path.exists('./Docs'):
            os.makedirs('./Docs')

        with open('./Docs/test_results.txt', 'a') as f:
            # Run Information
            f.write('--- Run Information ---\n')
            f.write(f'Test-run: {current_run_number}\n')
            f.write('\n')

            # Model Information
            f.write('--- Model Information ---\n')
            if os.path.isfile(self.model_handler.saved_model_path):
                f.write("Model: Previously saved model was used.\n")
                f.write(f'Previous accuracy: {self.get_previous_accuracy()}\n')
                f.write(f'Latest accuracy: {test_acc}\n')
            else:
                f.write("Model: New model was trained.\n")
            f.write(
                f'Regularization: {self.model_handler.regularization_method}, with factor: {self.model_handler.regularization_factor}\n\n')

            # Data Information
            f.write('--- Data Information ---\n')
            f.write(f'Train images: {self.data_handler.train_img_num}\n')
            f.write(f'Test images: {self.data_handler.test_img_num}\n')
            f.write(f'Batch size: {self.data_handler.batch_size}\n')
            f.write(f'Epochs run: {self.model_handler.epochs}\n')
            f.write(f'Early stopping patience: {self.model_handler.early_stopping.patience}\n\n')

            # Results
            f.write('--- Results ---\n')
            f.write(f'Correct guesses: {correct_guesses}/{self.data_handler.test_img_num}\n')
            f.write(f'Test accuracy: {test_acc}\n')
            f.write(f'Overall accuracy: {overall_accuracy:.3f}%\n')
            f.write(f'Improvement in accuracy: {test_acc - self.previous_accuracy}\n')

            if self.model_handler.early_stopping.stopped_epoch > 0:
                f.write(f'Training stopped early at epoch: {self.model_handler.early_stopping.stopped_epoch}\n')
            else:
                f.write("Training completed and did not stop early.\n\n\n")

        print("Test results saved to ./Docs/test_results.txt.")

    def plot_training_results(self):
        if not os.path.exists('./Plots/'):
            os.makedirs('./Plots/')


        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.model_handler.history.history['accuracy'])
        plt.plot(self.model_handler.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig("./Plots/acc_vals.jpg")

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(self.model_handler.history.history['loss'])
        plt.plot(self.model_handler.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig("./Plots/loss_vals.jpg")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))

        # Plot the training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.model_handler.history.history['loss'], label='Training Loss')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot the training accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.model_handler.history.history['accuracy'], label='Training Accuracy')
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

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
train_img_num = 8000
test_img_num = int(train_img_num * 0.2)
batch_size = 16
epochs = 200
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
patiance_value = 200
#############################################

data_handler = DataHandler(train_img_num, test_img_num, batch_size)
# Load Training and Testing Data
data_handler.load_training_and_test_data()

model_handler = ModelHandler(epochs, data_handler, timestamp, regularization_method, regularization_factor, patiance_value)
print(f"Running model with early-stopping; Patiance value set to {patiance_value}")
# Train Model
model_handler.handle_model_training()

# Test Model
tester = Tester(data_handler, model_handler, timestamp)
tester.test_model()
