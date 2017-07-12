import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class SupportVectorMachine:

	def __init__(self, visualisation=True):
		self.visualisation = visualisation
		self.colors = {1:'r', -1:'b'}
		if self.visualisation:
			self.fig = plt.figure()
			self.ax = self.fig.add_subplot(1, 1, 1)

	def fit(self, data):
		# training the SVM with data

		self.data = data
		
		# equation: ||w||: [w, b]
		opt_dict = {}

		# different step directions
		transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]

		# find min and max values of the features
		self.max_feature_value = -999999
		self.min_feature_value = 999999
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					if feature > self.max_feature_value:
						self.max_feature_value = feature
					if feature < self.min_feature_value:
						self.min_feature_value = feature

		step_sizes = [self.max_feature_value * 0.1, self.max_feature_value * 0.01, self.max_feature_value * 0.001]

		b_range_multiple = 5 # b is very expensive
		b_multiple = 5 # larger steps okay with b, in comparison to w

		latest_optimum = self.max_feature_value*10

		# processing and classifying all the data
		for step in step_sizes:
			w = np.array([latest_optimum, latest_optimum])
			optimized = False
			while not optimized:
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value*b_range_multiple, step*b_multiple):
					for transformation in transforms:
						w_t = w*transformation
						found_option = True
						# equation: yi(xi.w+b) >= 1
						for i in self.data: # god damn this is expensive, need to optimize
							for xi in self.data[i]:
								yi=i
								if not yi*(np.dot(w_t, xi)+b) >= 1:
									found_option = False
									break

						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t,b] # magnitude of the vector

					if w[0] < 0:
						optimized = True
						print("Optimized: w ({}) is smaller than 0 ".format(w[0]))
					else:
						w = w - step # yes, this is mixing a vector with a scalar, but it works

				norms = sorted([n for n in opt_dict])
				opt_choice = opt_dict[norms[0]]

				self.w = opt_choice[0]
				self.b = opt_choice[1]

				print("w={} , b={}".format(self.w, self.b))

				latest_optimum = opt_choice[0][0]+step*2

	def predict(self, features):
		# equation: sign(x.w+b)
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		if classification != 0 and self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker="*", c=self.colors[classification])

		return classification

	def visualize(self):
		# just for the user, doesn't touch the svm

		[[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

		# hyperplane: x.w+b
		def hyperplane(x, w, b, v):
			return (-w[0]*x-b+v) / w[1]

		datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
		hyp_x_min = datarange[0]
		hyp_x_max = datarange[1]

		# positive support vector
		psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
		psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
		self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'r')

		# negative support vector
		nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
		nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
		self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'b')

		# decision boundry
		db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
		db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
		self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

		plt.show()

data_dict = {-1: np.array([[1,7], [2,8], [3,8]]),
			  1: np.array([[5,1], [6,-1], [7,3]])}

clf = SupportVectorMachine()
clf.fit(data=data_dict)
clf.visualize()