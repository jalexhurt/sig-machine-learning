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

#get cutoff for folds
num_samples = len(data)
cutoff = num_samples // 2

# create folds
foldA_train, foldB_train = data[:cutoff], data[cutoff:]
foldA_train_labels, foldB_train_labels = classes[:cutoff], classes[cutoff:]
foldA_test, foldB_test = foldB_train, foldA_train
foldA_test_labels, foldB_test_labels = foldB_train_labels, foldA_train_labels

# dimension of data and number of classes
dimension = 4
num_classes = 3
#####################################


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=dimension))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))



###########FOLD A###############
from keras.models import clone_model
model_A = clone_model(model)
model_A.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_A.fit(foldA_train, foldA_train_labels, epochs=20)

loss, accuracy1 = model_A.evaluate(foldA_test, foldA_test_labels)



###########FOLD B###############
from keras.models import clone_model
model_B = clone_model(model)
model_B.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_B.fit(foldB_train, foldB_train_labels, epochs=20)

loss, accuracy2 = model_A.evaluate(foldB_test, foldB_test_labels)


####################################
print("Accuracy on fold A: {}".format(accuracy1))
print("Accuracy on fold B: {}".format(accuracy2))



