from sklearn.datasets import load_iris
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

iris = load_iris()
data, classes = iris.data, iris.target
classes = keras.utils.to_categorical(classes, num_classes = 3)

# randomize data
p = numpy.random.permutation(len(data))
data = data[p]
classes = classes[p]

dimension = 4
num_classes = 3

#####################################


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=dimension))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, classes, epochs=20)



