import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import random
from matplotlib import style
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

style.use('ggplot')

class Eye:

	def __init__(self):
		print("Eye: Inititiated")

	def look(self, x, y):
		potential_cells = []

		for cell in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
			potential_cells.append((x+cell[0], y+cell[1]))

		return potential_cells

	def percieve_area(self, agents, x, y):
		x_offset = x-3
		y_offset = y-3

		agent_list = []
		agent_pos_list = []

		for xx in range(0, 7):
			for yy in range(0, 7):

				xx1 = xx + x_offset
				yy1 = yy + y_offset

				for agent in agents:
					if agent[4] == xx1 and agent[5] == yy1: 
						agent_list.append(agent[0])
						agent_pos_list.append((xx, yy))

		return agent_list, agent_pos_list

class Brain:

	def __init__(self, k=3, *args, **kwargs):
		self.k = k
		self.clf = KNeighborsClassifier(n_neighbors=self.k, algorithm="ball_tree", weights="uniform", n_jobs=1)
		self.X = np.array([])
		self.y = np.array([])

		print("Brain: Inititiated")

	def process_experience(self, X, sentiment, details=False):
		self.X = self.X.tolist()
		self.y = self.y.tolist()

		self.X.append(X)
		self.y.append(sentiment)

		self.X = np.array(self.X)
		self.y = np.array(self.y)

		if details:
			print("Brain: Experience processed! sentiment={}".format(sentiment))

	def add_data(self, X_new, y_new, details=False):
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

		if details:
			print("Brain: Added {} entries of data".format(len(y_new)))

	def total_experiences(self):
		return len(self.y)

	def learn(self, details=False):
		start_time = time.time()

		self.clf.fit(self.X, self.y)

		if details:
			print("Brain: Fitted {} samples [{}s]".format(len(self.y),round(time.time() - start_time,5)))

	def predict(self, X):
		return self.clf.predict(X)[0]

	def evaluate_agents(self, agents, agent_list):
		agent_sent_list = []

		for i in agent_list:
			try:
				agent_sent_list.append(self.predict([[agents[i][2], agents[i][3], -1]]))
			except:
				pass

		return agent_sent_list

		

	def visualize(self, title="", show_text=True, mesh_step_size=0.1, show=True, time_limit=3600):
		# step size in the mesh

		start_time = time.time()

		x_min, x_max = self.X[:, 0].min() - 0.3, self.X[:, 0].max() + 0.3
		y_min, y_max = self.X[:, 1].min() - 0.3, self.X[:, 1].max() + 0.3
		xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))

		# plot setup
		plt.figure(figsize=(6,6))
		# cm = ListedColormap("#FFA0A0 #FFB8B8 #FFB0B0 #FFC8C8 #FFC0C0 #FFD0D0 #FFFFFF #E0FFE0 #D0FFD0 #D8FFD8 #C0FFC0 #C8FFC8 #B0FFB0".split())
		cm = ListedColormap("#FFA0A0 #FFB0B0 #FFD0D0 #EEEEEE #C0FFC0 #B0FFB0 #A0FFA0".split())
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
		ax.pcolormesh(xx, yy, self.Z, cmap=cm, alpha=1)
		# ax.contourf(xx, yy, self.Z, cmap=cm, alpha=1)

		# plot X as scatter
		ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cm_bold, marker="o", alpha=0.05)
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(np.arange(3))
		ax.set_yticks(np.arange(3))

		plt.gca().invert_yaxis()
		# ax.set_xlabel("X1")
		# ax.set_ylabel("X2")

		dimensions = np.shape(self.X)[1]

		ax.set_title("{} [2/{} dimensions]".format(title, dimensions))

		if show_text:
			# ax.text(xx.min() + 0.05, yy.max() - 0.1, "2/{} dimensions".format(dimensions), size=12, horizontalalignment='left')
			ax.text(xx.min() + 0.05, yy.min() + 0.05, "n_samples={}".format(len(self.y)), size=12, horizontalalignment='left')
			ax.text(xx.max() - 0.05, yy.min() + 0.05, "k={}".format(self.k), size=12, horizontalalignment='right')

		print("Brain: Prepared visualization [{}s]".format(round(time.time() - start_time,5)))

		start_show_time = time.time()

		if show:
			plt.ion()
			plt.pause(time_limit)
			plt.close()

		# return plot
		return plt



class Muscle:

	def __init__(self):
		print("Muscle: Inititiated")


	def move(self, x, y):
		return x, y


if __name__ == "__main__":
	brain = Brain()

	X_new, y_new = make_moons(n_samples=100, noise=0.5, random_state=2)
	brain.add_data(X_new, y_new)

	brain.learn()
	plt = brain.visualize()
	plt.show()