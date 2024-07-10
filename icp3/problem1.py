import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(155)

# load dataset
dataset = pd.read_csv("breastcancer.csv").drop(columns=["id", "Unnamed: 32"])
X_train, X_test, Y_train, Y_test = train_test_split(
    StandardScaler().fit_transform(dataset.drop(columns=["diagnosis"])),
    (dataset["diagnosis"] == "M").astype("int"),
    test_size=0.25,
    random_state=87,
)

my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(20, input_dim=30, activation="relu"))  # hidden layer
my_first_nn.add(Dense(20, activation="relu"))
my_first_nn.add(Dense(20, activation="relu"))
my_first_nn.add(Dense(20, activation="relu"))
my_first_nn.add(Dense(20, activation="relu"))
my_first_nn.add(Dense(1, activation="sigmoid"))  # output layer
my_first_nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)

print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))

# 1. Adding 4 additional hidden layers increased accuracy by about 0.03 and reduced loss by about 0.1
# 2. Breast cancer model accuracy is 0.8456
# 3. Adding the StandardScaler increases the accuracy to 0.9650
