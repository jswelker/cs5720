import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape[1:])
# process the data
# 1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
print(dimData)
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# convert data to float and scale values between 0 and 1
train_data = train_data.astype("float")
test_data = test_data.astype("float")
# scale data
train_data /= 255.0
test_data /= 255.0
# change the labels frominteger to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# creating network
model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(dimData,)))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    train_data,
    train_labels_one_hot,
    batch_size=256,
    epochs=10,
    verbose=1,
    validation_data=(test_data, test_labels_one_hot),
)

# 1. Plot the accuracy and loss
df = pd.DataFrame(history.history)
training_plot = plt.scatter(df["accuracy"], df["loss"], color="blue")
validation_plot = plt.scatter(df["val_accuracy"], df["val_loss"], color="green")
plt.show()

# 2. Plot one test image and do an inference check
plt.imshow(test_data[0].reshape(28, 28))
plt.show()
predictions = model.predict(test_data[0].reshape(1, 784))
print(f"Predicted category is: {predictions[0].argmax()}")

# 3. Changing Relu to Sigmoid and adding 2 layers brings accuracy much lower and loss much higher
# Original
# {'accuracy': [0.9106000065803528, 0.9700666666030884, 0.98089998960495, 0.9856500029563904, 0.9898166656494141, 0.9935666918754578, 0.9948833584785461, 0.9956166744232178, 0.9967833161354065, 0.9977499842643738], 'loss': [0.29241472482681274, 0.09874562919139862, 0.06293473392724991, 0.044316887855529785, 0.03197728469967842, 0.02188359573483467, 0.01569233275949955, 0.013509051874279976, 0.009321410208940506, 0.007093319669365883], 'val_accuracy': [0.9419999718666077, 0.9685999751091003, 0.9764999747276306, 0.9782000184059143, 0.9794999957084656, 0.9807000160217285, 0.9789999723434448, 0.9829999804496765, 0.9829000234603882, 0.9786999821662903], 'val_loss': [0.1842980533838272, 0.09574449807405472, 0.07435309141874313, 0.06727549433708191, 0.07155095785856247, 0.0689258947968483, 0.07561755925416946, 0.06487123668193817, 0.06615250557661057, 0.09180568158626556]}
# New
# {'accuracy': [0.2595166563987732, 0.7334666848182678, 0.8479666709899902, 0.8864166736602783, 0.9129999876022339, 0.9310833215713501, 0.9432666897773743, 0.9527999758720398, 0.9584833383560181, 0.9616666436195374], 'loss': [1.9893066883087158, 0.772314190864563, 0.4853484630584717, 0.3692099153995514, 0.28331395983695984, 0.2239917665719986, 0.18414010107517242, 0.15499012172222137, 0.13648621737957, 0.12430287897586823], 'val_accuracy': [0.6074000000953674, 0.8324999809265137, 0.8723999857902527, 0.8912000060081482, 0.9179999828338623, 0.9164000153541565, 0.9409000277519226, 0.9375, 0.9182000160217285, 0.9496999979019165], 'val_loss': [0.9901183843612671, 0.5572165250778198, 0.413941890001297, 0.3543725609779358, 0.26631736755371094, 0.2626403272151947, 0.1934795081615448, 0.20109917223453522, 0.27435147762298584, 0.1617741733789444]}

# 4. Removing the scaling step brings accuracy a small bit lower and loss much higher
# Original
# {'accuracy': [0.9106000065803528, 0.9700666666030884, 0.98089998960495, 0.9856500029563904, 0.9898166656494141, 0.9935666918754578, 0.9948833584785461, 0.9956166744232178, 0.9967833161354065, 0.9977499842643738], 'loss': [0.29241472482681274, 0.09874562919139862, 0.06293473392724991, 0.044316887855529785, 0.03197728469967842, 0.02188359573483467, 0.01569233275949955, 0.013509051874279976, 0.009321410208940506, 0.007093319669365883], 'val_accuracy': [0.9419999718666077, 0.9685999751091003, 0.9764999747276306, 0.9782000184059143, 0.9794999957084656, 0.9807000160217285, 0.9789999723434448, 0.9829999804496765, 0.9829000234603882, 0.9786999821662903], 'val_loss': [0.1842980533838272, 0.09574449807405472, 0.07435309141874313, 0.06727549433708191, 0.07155095785856247, 0.0689258947968483, 0.07561755925416946, 0.06487123668193817, 0.06615250557661057, 0.09180568158626556]}
# New
# {'accuracy': [0.8809499740600586, 0.9461333155632019, 0.9611333608627319, 0.9666500091552734, 0.9732999801635742, 0.9755499958992004, 0.9785000085830688, 0.980566680431366, 0.9834333062171936, 0.9847000241279602], 'loss': [5.0903520584106445, 0.4126495122909546, 0.24992598593235016, 0.19452591240406036, 0.16123487055301666, 0.1603706330060959, 0.14813947677612305, 0.13399846851825714, 0.12386826425790787, 0.11971642076969147], 'val_accuracy': [0.8892999887466431, 0.953000009059906, 0.9535999894142151, 0.955299973487854, 0.9537000060081482, 0.9606999754905701, 0.9657999873161316, 0.9635999798774719, 0.9714000225067139, 0.9726999998092651], 'val_loss': [1.077016830444336, 0.37685859203338623, 0.4137918949127197, 0.31378909945487976, 0.34294572472572327, 0.4647425413131714, 0.373721718788147, 0.424380362033844, 0.32395994663238525, 0.3264416754245758]}
