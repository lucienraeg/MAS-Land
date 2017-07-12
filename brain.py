import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

style.use('ggplot')

random_state = 1

# classifier parameters
k = 3

# step size in the mesh
h = 0.02

class Classifier:

	def __init__(self, testing=True):
		self.clf = KNeighborsClassifier(k)
		self.testing = testing

	def fitData(self, X, y):
		self.X = X
		self.y = y

		self.X = StandardScaler().fit_transform(X)

		if self.testing:
			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
		else:
			self.X_train = self.X
			self.y_train = self.y

		self.clf.fit(self.X_train, self.y_train)

		if self.testing:
			self.score = accuracy_score(self.clf.predict(self.X_test), self.y_test)
			print("Accuracy Score: {}".format(self.score))

	def predict(self, X):
		y_predicted = self.clf.predict(X)

		if self.testing:
			print("Predicted: {}".format(label_dict[y_predicted[0]]))

		return y_predicted

	def predictProba(self, X):
		y_predicted = self.clf.predict(X)
		y_proba = self.clf.predict_proba(X)
		y_proba = y_proba[0].tolist()

		if self.testing:
			print("Predicted: {}".format(label_dict[y_predicted[0]]))
			y_proba_full = []
			for i, proba in enumerate(y_proba):
				print("Probability of {}: {}".format(label_dict[i], round(proba,2)))

		return y_predicted, y_proba


X = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

label_dict = {0: "red", 1: "blue"}

classifier = Classifier()
classifier.fitData(X, y)

classifier.predictProba([[0, 0]])