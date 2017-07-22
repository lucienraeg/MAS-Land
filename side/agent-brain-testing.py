import numpy as np
import matplotlib.pyplot as plt
import time
import random
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, load_iris
from sklearn.neighbors import KNeighborsClassifier

style.use('ggplot')

class Eye:

	def __init__(self):
		print("Eye: Inititiated")

	def look(self, x, y):
		potential_cells = []

		for cell in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
			potential_cells.append((x+cell[0], y+cell[1]))

		# move_pos = random.choice(potential_cells)

		return potential_cells

class Brain:

	def __init__(self, k=3, *args, **kwargs):
		self.k = k
		self.clf = KNeighborsClassifier(self.k)
		self.X = np.array([])
		self.y = np.array([])

		print("Brain: Inititiated")

	def process_experience(self, X_1, X_2, sentiment):
		self.X = self.X.tolist()
		self.y = self.y.tolist()

		self.X.append([X_1, X_2])
		self.y.append(sentiment)

		self.X = np.array(self.X)
		self.y = np.array(self.y)

		print("Brain: Experience processed! sentiment={}".format(sentiment))

	def add_data(self, X_new, y_new):
		# turn into list
		self.X = self.X.tolist()
		self.y = self.y.tolist()

		# add new data
		for X_entry in X_new:
			self.X.append([X_entry[0], X_entry[1]])

		for y_entry in y_new:
			self.y.append(y_entry)

		# turn into numpy array
		self.X = np.array(self.X)
		self.y = np.array(self.y)

		print("Brain: Added {} entries of data".format(len(y_new)))

	def total_experiences(self):
		return len(self.y)

	def learn(self):
		start_time = time.time()

		self.clf.fit(self.X, self.y)

		print("Brain: Fitted {} samples [{}s]".format(len(self.y),round(time.time() - start_time,5)))

	def predict(self, X):
		return(self.clf.predict(X))

	def visualize(self, title="", show_text=True, mesh_step_size=0.1):
		# step size in the mesh

		start_time = time.time()

		x_min, x_max = self.X[:, 0].min() - 0.3, self.X[:, 0].max() + 0.3
		y_min, y_max = self.X[:, 1].min() - 0.3, self.X[:, 1].max() + 0.3
		xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))

		# plot setup
		plt.figure(figsize=(6,6))
		cm_light = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
		cm_bold = ListedColormap(["#000000"])
		ax = plt.subplot(1, 1, 1)

		print("Brain: Set up figure [{}s]".format(round(time.time() - start_time,5)))
		start_time = time.time()

		# get predictions
		self.Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
		self.Z = self.Z.reshape(xx.shape)

		print("Brain: Calculated predictions ({}) [{}s]".format(len(xx)*len(yy), round(time.time() - start_time,5)))
		start_time = time.time()

		# plot Z as contour
		# ax.pcolormesh(xx, yy, self.Z, cmap=cm, alpha=1)
		ax.contourf(xx, yy, self.Z, c=self.Z, cmap=cm_light, alpha=1)

		# plot X as scatter
		ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cm_bold, marker="o", alpha=0.05)
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		# ax.set_xticks(())
		# ax.set_yticks(())
		ax.set_xlabel("Color")
		ax.set_ylabel("Shape")


		ax.set_title(title)

		if show_text:
			ax.text(xx.min() + 0.1, yy.min() + 0.1, "n_samples={}".format(len(self.y)), size=12, horizontalalignment='left')
			ax.text(xx.max() - 0.1, yy.min() + 0.1, "k={}".format(self.k), size=12, horizontalalignment='right')

		print("Brain: Prepared visualization [{}s]".format(round(time.time() - start_time,5)))

		print(self.y)

		# show graph
		plt.show()



class Muscle:

	def __init__(self):
		print("Muscle: Inititiated")


	def move(self, x, y):
		return x, y


if __name__ == "__main__":
	brain = Brain()

	iris = load_iris()

	X_new, y_new = iris.data[:, :2], iris.target
	brain.add_data(X_new, y_new)

	brain.learn()
	brain.visualize()