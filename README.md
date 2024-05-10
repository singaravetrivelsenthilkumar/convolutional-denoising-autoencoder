# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Develop a convolutional autoencoder for image denoising application. Given noisy images as input, the objective is to reconstruct clean images while removing the noise.

Dataset:

The MNIST dataset, a collection of handwritten digits, is used for this task. It consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels in grayscale.

## Convolution Autoencoder Network Model

![image](https://github.com/DHARINIPV/convolutional-denoising-autoencoder/assets/119400845/2600fce8-b5ad-4207-98e7-4b265d7f3d15)


## DESIGN STEPS

### STEP 1:
Load and preprocess the MNIST dataset, normalize pixel values, and add random noise to images.

### STEP 2:
Display a subset of noisy images from the test set in a 2-row, 5-column grid.

### STEP 3:
Define an autoencoder model using the Functional API of Keras, including encoder and decoder components.

### STEP 4:
Train the autoencoder model using the noisy training images, optimizing with the Adam optimizer and binary cross-entropy loss, for a specified number of epochs.

### STEP 5:
Reconstruct noisy images from the test set using the trained autoencoder, and display original images, corresponding noisy images, and their reconstructions in a 3-row, 10-column grid.


## PROGRAM
```
Developed by: singaravetrivel S
Register No: 212222220048
```
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))
noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape) 
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(2, 5, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
input_img = keras.Input(shape=(28, 28, 1))

# Write your encoder here
x = layers.Conv2D(16, (3,3), activation = 'relu', padding='same') (input_img)
x =layers.MaxPooling2D((2,2), padding='same') (x)
x = layers.Conv2D(8, (3,3), activation = 'relu', padding='same') (x)
x = layers. MaxPooling2D((2,2), padding='same') (x)
x =layers.Conv2D(8, (3,3), activation = 'relu', padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Encoder output dimension is ## Mention the dimention ##

x = layers.Conv2D(8, (3,3), activation = 'relu',padding='same') (encoded)
x = layers. UpSampling2D((2,2))(x)
x = layers.Conv2D(8, (3,3),activation = 'relu', padding='same') (x)
x = layers. UpSampling2D((2,2))(x)
x = layers.Conv2D(16, (3,3), activation = 'relu')(x)
x = layers. UpSampling2D((2,2))(x)

# Write your decoder here

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
decoded_imgs = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot from 2024-04-26 23-47-10](https://github.com/DHARINIPV/convolutional-denoising-autoencoder/assets/119400845/103c8d7b-597c-4afd-8ca2-5da54b4524a8)

### Original vs Noisy Vs Reconstructed Image

![Screenshot from 2024-04-26 23-47-46](https://github.com/DHARINIPV/convolutional-denoising-autoencoder/assets/119400845/76d5b26e-0727-43ed-adc3-5fc30aea3550)


## RESULT
Thus, a convolutional autoencoder for image denoising application is developed.
