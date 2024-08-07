# Simple CNN model for CIFAR-10
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from keras.constraints import MaxNorm
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

# from keras import backend as K
# K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
model.add(
    Conv2D(32, (3, 3), activation="relu", padding="same", kernel_constraint=MaxNorm(3))
)
model.add(Dropout(0.2))
model.add(
    Conv2D(32, (3, 3), activation="relu", padding="same", kernel_constraint=MaxNorm(3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(64, (3, 3), activation="relu", padding="same", kernel_constraint=MaxNorm(3))
)
model.add(Dropout(0.2))
model.add(
    Conv2D(64, (3, 3), activation="relu", padding="same", kernel_constraint=MaxNorm(3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(128, (3, 3), activation="relu", padding="same", kernel_constraint=MaxNorm(3))
)
model.add(Dropout(0.2))
model.add(
    Conv2D(128, (3, 3), activation="relu", padding="same", kernel_constraint=MaxNorm(3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation="relu", kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu", kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# Compile model
epochs = 25
lrate = 0.01
sgd = SGD(learning_rate=lrate, momentum=0.9, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Fit the model
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32
)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

predictions = model.predict(X_test[0:4])
print("First four predicted digits are: " + [p.argmax() for p in predictions])

df = pd.DataFrame(history.history)
plt.scatter(df["accuracy"], df["loss"])
plt.show()

# 1. The accuracy did improve by adding the additional layers.
# Original output: Accuracy: 61.53%
# New output: Accuracy: 76.36%

# 2. The predicted digits are 5,8,8,0

# 3. See screenshots in submission.
