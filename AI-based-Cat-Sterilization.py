# Import necessary libraries
import tensorflow as tf
import os
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.python import keras
from tensorflow.python.keras import layers, regularizers, optimizer_v1
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Disable eager execution mode
tf.compat.v1.disable_eager_execution()

# Set image size and hyperparameters
resize = 224
weight_decay = 0.0005
nb_epoch = 100
batch_size = 8


# Load data
def load_data():
    imgs = os.listdir(path)  # Get all file names in the path
    num = len(imgs)
    data = []  # Store image data
    label = []  # Store labels
    for img in imgs:
        try:
            # Get image category
            category = img.split(".")[0]
            img_path = os.path.join(path, img)  # Get image path
            img = cv2.imread(img_path)  # Read image
            img = cv2.resize(img, (resize, resize))  # Resize image to resize * resize
            img = img / 255.0  # Normalize
            data.append(img)
            
            # Modify labels for the new classification problem
            if category == "cat0":
                # Assuming cat without ear cropping (label: 0)
                label.append("Cat without ear cropping")
            elif category == "cat1":
                # Assuming cat with ear cropping (label: 1)
                label.append("Cat with ear cropping")
        except:
            print("Error: ", img)
    data = np.array(data)  # Convert data to numpy array format
    label = np.array(label)  # Convert labels to numpy array format
    return data, label
train_data, train_label = load_data()
print("Train data shape: ", train_data.shape)
print("Train label shape: ", train_label.shape)


# Dataset visualization based on modified labels
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    if train_label[i] == "Cat without ear cropping":
        plt.xlabel("Cat without ear cropping")
    elif train_label[i] == "Cat with ear cropping":
        plt.xlabel("Cat with ear cropping")
plt.show()

# Convert labels for the new classification problem (0: cat without ear cropping, 1: cat with ear cropping) 
# Assuming: "Cat without ear cropping" label as 0, "Cat with ear cropping" label as 1
train_label_new = np.where(train_label == "Cat without ear cropping", 0, 1)
train_label_new = to_categorical(train_label_new, 2)  # Convert to one-hot encoding format


# Define VGG16 model
def vgg160():
    # Initialize the model architecture
    model = keras.Sequential()
    
    # Adding the first convolutional layer
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(224, 224, 3), 
                            kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    # Adding the second convolutional layer
    model.add( layers.Conv2D( 64, (3, 3), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Adding the third convolutional layer
    model.add(layers.Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the forth convolutional layer
    model.add(layers.Conv2D(128,(3,3), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    # Adding the fifth convolutional layer
    model.add(layers.Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the sixth convolutional layer
    model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the seventh convolutional layer
    model.add(layers.Conv2D(256,(3,3),padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    # Adding the eighth convolutional layer
    model.add(layers.Conv2D(512,(3,3), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the ninth convolutional layer
    model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the tenth convolutional layer
    model.add(layers.Conv2D(512,(3,3),padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    # Adding the eleventh convolutional layer
    model.add(layers.Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the twelfth convolutional layer
    model.add(layers.Conv2D(512,(3,3),padding='same', kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    # Adding the thirteenth convolutional layer
    model.add(layers.Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.5))
    # Adding the fourteenth convolutional layer
    model.add(layers.Flatten())
    model.add(layers.Dense(512,kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    # Adding the fifteenth convolutional layer
    model.add(layers.Dense(512,kernel_regularizer=regularizers.l1(weight_decay)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    # Adding the sixteenth convolutional layer
    model.add(layers.Dropout(0.8))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    return model

    model.add(layers.Dense(3, activation='softmax'))  # Add a neuron for the new classification problem
    return model

# Build VGG16 model (Update the model name to reflect its purpose)
model = load_model('vggnet-16_cat_ear_crop.h5')  # Update the model filename reflecting its purpose

model.layers[-1].activation = layers.Activation('softmax')

# Modify the last layer of the model
model = vgg160()

# Print model summary
model.summary()

# Define optimizer
sgd = optimizer_v1.SGD(lr=1e-9, momentum=0.9, decay=1e-6, nesterov=True)
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Train the model
history = model.fit(train_data, train_label_new, batch_size=4, epochs=nb_epoch, validation_split=0.2, shuffle=True)

# Plot accuracy for training and validation sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)