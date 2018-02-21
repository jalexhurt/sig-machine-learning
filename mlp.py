

###########
# Imports
###########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Dense

plotting = True
################
# Data Setup
###############
X = np.array([[1,2], [1,0], [1,1], [1.5,1], [-.2, -.5], [-.5, 0], [-.25,0]])
y = np.array([0,0,0,0,1,1,1])
test_x = np.array([[0,0]])

Y = np.zeros([len(y), len(X[0])])

for i, sample in enumerate(Y):
    sample[y[i]] = 1

###########
# MLP
##########
model = Sequential()
model.add(Dense(3, input_shape=(2,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=['accuracy']) 
model.fit(X,Y, epochs=5)

pred = model.predict(test_x)

print(f"Predictions is {pred}")

##################
# Plotting
#################
if plotting:
    for pt, label in zip(X,y):
        plt.scatter(pt[0], pt[1], c='red' if label == 0 else 'blue')
    
red_patch = mpatches.Patch(color='red', label='Class 0')
blue_patch = mpatches.Patch(color='blue', label='Class 1')
black_patch = mpatches.Patch(color='black', label='Test Point')

plt.scatter(test_x[0,0], test_x[0,1], c='black')
plt.legend(handles=[red_patch, blue_patch, black_patch])
plt.show()
