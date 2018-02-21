###########
# Imports
###########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier as KNN

plotting = True

################
# Data Setup
###############
X = np.array([[1,2], [1,0], [1,1], [1.5,1], [-.2, -.5], [-.5, 0], [-.25,0]])
y = np.array([0,0,0,0,1,1,1])
n_neighbors = 6
test_x = np.array([[0,0]])

###########
# KNN
##########

# we create an instance of Neighbours Classifier and fit the data.
knn = KNN(n_neighbors)
knn.fit(X, y)
pred = knn.predict(test_x)

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
