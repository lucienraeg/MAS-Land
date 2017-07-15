import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.datasets import make_classification

style.use("ggplot")

# classifier parameters
n_neighbors = 3

# data
# X = np.array([[0, 0,], [0, 1], [0.5, 3], [1, 3.5], [4, 4.8], [5, 5], [7, 5], [7, 7]])
# y = np.array([0, 0, 1, 1, 2, 2, 3, 3])

X, y = make_classification(n_samples=50, n_features=4, n_redundant=0, n_informative=4, n_classes=4, random_state=1, n_clusters_per_class=2)

X = X[:, :2]

h = 0.02  # step size in the mesh

# Create color maps
cmap_back = ListedColormap(["#FFBBBB", "#FFDDDD", "#DDFFDD", "#BBFFBB"])
cmap_front = ListedColormap(["#000000"])

# create and fit KNN
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]*[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_back)
# plt.contourf(xx, yy, Z, cmap=cmap_back, alpha=1)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_front, marker="x")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()