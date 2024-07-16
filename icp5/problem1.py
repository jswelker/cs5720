# Auto-encoder
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist, mnist
from keras.layers import Dense, Input
from keras.models import Model, Sequential

# this is the size of our encoded representations
encoding_dim = (
    32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
)

# this is our input placeholder
input_img = Input(shape=(784,))
encoded = Dense(512, activation="relu")(input_img)
encoded = Dense(256, activation="relu")(encoded)
encoded = Dense(128, activation="relu")(encoded)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(256, activation="relu")(decoded)
decoded = Dense(512, activation="relu")(decoded)
decoded = Dense(784, activation="sigmoid")(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
autoencoder.compile(optimizer="adam", loss="binary_focal_crossentropy")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=25,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)
output = autoencoder.predict(x_test)

print("before")
plt.imshow(x_test[0].reshape(28, 28))

print("after")
plt.imshow(output[0].reshape(28, 28))
