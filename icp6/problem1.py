# This will only work with keras==2.12.0

import re

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("sentiment.csv")
# Keeping only the neccessary columns
data = data[["text", "sentiment"]]

data["text"] = data["text"].apply(lambda x: x.lower())
data["text"] = data["text"].apply((lambda x: re.sub("[^a-zA-z0-9\s]", "", x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace("rt", " ")

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(data["text"].values)
X = tokenizer.texts_to_sequences(data["text"].values)

X = pad_sequences(X, value=1)

embed_dim = 128
lstm_out = 196


def createmodel():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data["sentiment"])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

param_grid = {"epochs": [10, 20], "batch_size": [16, 32, 64]}

model = KerasClassifier(build_fn=createmodel)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, Y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)

# TODO: I cannot find a way to save the output of the trained GridSearchCV object
new_phrases = ["I hate pickles!", "I like puppies."]
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(new_phrases)
X_new = tokenizer.texts_to_sequences(new_phrases)
X_new = pad_sequences(X_new, 28)
print(grid.predict(X_new))
