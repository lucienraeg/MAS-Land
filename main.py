import pygame
import random
import csv
import agent
import numpy as np
import sklearn

names = []

with open("agent-names.csv", "rt") as f:
	reader = csv.reader(f)
	for row in reader:
		names.append(row[0])

seed = 224225
running = True

random.seed(seed)

class Window:

	def __init__(self):
		pygame.init()

		# init display
		self.display_width = 32*30
		self.display_height = 32*30
		self.side_display_width = 32*20
		self.side_display_height = 32*0
		self.display = pygame.display.set_mode((self.display_width+self.side_display_width, self.display_height+self.side_display_height))

		# init misc
		self.clock = pygame.time.Clock()
		self.world_speed = 20
		self.grid_size = 32
		self.total_steps = 0

		# init focus
		self.focus = None # default = None
		self.focus_visualize_frequency = 5 # how many steps between
		self.focus_visualize_time = 1 # in seconds

		# init colors
		self.WHITE = (255, 255, 255)
		self.GRAY = (225, 225, 225)
		self.BLACK = (0, 0, 0)
		self.RED = (255, 0, 0)
		self.GREEN = (0, 255, 0)
		self.BLUE = (0, 0, 255)
		self.YELLOW = (255, 255, 0)

		self.LT_RED = (255, 50, 50)
		self.LT_GREEN = (50, 255, 50)

		# init font
		self.FNT_SMALL = pygame.font.SysFont("arial", 11)
		self.FNT_MEDIUM = pygame.font.SysFont("arial", 14)

		# init dicts
		self.colors = {0: self.RED, 1: self.GREEN, 2: self.BLUE}
		self.shapes = {0: "square", 1: "circle", 2: "triangle"}
		self.sentiments = {-1: "negative", 0: "neutral", 1: "positive"}
		self.sentiment_colors = {-1: self.LT_RED, 0: self.GRAY, 1: self.LT_GREEN}

		# init agents
		starting_agents = 32
		self.agents = []
		for i in range(starting_agents):
			self.create_agent(i)

	def main(self):
		self.display.fill(self.WHITE)

		self.draw_grid(self.grid_size)

		agent_pos_list = []
		for agent in self.agents:
			agent_pos_list.append((agent[4], agent[5]))

		for agent in self.agents:
			eye = agent[6]
			brain = agent[7]
			muscle = agent[8]

			# process any experience had
			agent_contacts = []
			for a in enumerate(agent_pos_list):
				# check for contact with another agent
				if a[1] == (agent[4], agent[5]) and a[0] != agent[0]:
					other_agent = a[0]

					other_color = self.agents[other_agent][2]
					other_shape = self.agents[other_agent][3]
					other_x = self.agents[other_agent][4]
					other_y = self.agents[other_agent][5]

					# get features
					X_1 = other_color
					X_2 = other_shape

					# decide sentiment (label)
					if other_color == 0 and other_shape == 0:
						sent = -1
					elif other_color == 2:
						sent = 1
					else:
						sent = 0

					# experience
					self.experience(agent, [X_1, X_2], sent)

					# print experience
					if agent[0] == self.focus:
						print("[AGENT#{}] Experience w/ #{}! sentiment={}".format(agent[0], other_agent, self.sentiments[sent]))

					# learn from experiences
					if brain.total_experiences() > 3:
						brain.learn()

						if agent[0] == self.focus and brain.total_experiences() % self.focus_visualize_frequency == 0:
							# self.check_agent(agent[0])
							brain.visualize("AGENT#{}: {}".format(agent[0], agent[1]), time_limit=self.focus_visualize_time)
	
			# percieve area
			potential_cells = eye.look(agent[4], agent[5])

			# calculate best cell to move to
			move_cell = random.choice(potential_cells)

			# move to that cell
			agent[4], agent[5] = muscle.move(move_cell[0], move_cell[1])

			# clamp pos
			agent[4] = max(1, min(agent[4], (self.display_width//self.grid_size)-2))
			agent[5] = max(1, min(agent[5], (self.display_height//self.grid_size)-2))

		for agent in self.agents:
			# draw agent
			self.draw_agent(agent[4], agent[5], agent[2], agent[3], agent[0])

		self.display_sidebar(self.display_width+16, 8)

		pygame.display.update()
		self.clock.tick(self.world_speed)
		self.total_steps += 1

		pygame.display.set_caption("Seed: {}, FPS: {}".format(seed, round(self.clock.get_fps(),2)))

	def experience(self, agent, X, sentiment):
		agent[7].process_experience(X, sentiment)

	def draw_grid(self, grid_size):
		for col in range((self.display_width//grid_size)+1):
			pygame.draw.line(self.display, self.GRAY, (col*grid_size, 0), (col*grid_size, self.display_height))

		for row in range((self.display_height//grid_size)+1):
			pygame.draw.line(self.display, self.GRAY, (0, row*grid_size), (self.display_width, row*grid_size))

	def draw_agent(self, x, y, color, shape, number):
		x = x*self.grid_size+(self.grid_size//2)
		y = y*self.grid_size+(self.grid_size//2)

		if self.shapes[shape] == "square":
			pygame.draw.rect(self.display, self.colors[color], (x-10, y-10, 20, 20))
			pygame.draw.rect(self.display, self.BLACK, (x-11, y-11, 22, 22), 3)
		elif self.shapes[shape] == "circle":
			pygame.draw.circle(self.display, self.colors[color], (x, y), 12)
			pygame.draw.circle(self.display, self.BLACK, (x, y), 13, 3)
		elif self.shapes[shape] == "triangle":
			pygame.draw.polygon(self.display, self.colors[color], ((x, y-10), (x-10, y+10), (x+10, y+10)))
			pygame.draw.polygon(self.display, self.BLACK, ((x, y-12), (x-12, y+12), (x+12, y+12)), 3)

		num = self.FNT_SMALL.render("#{}".format(number), True, self.BLACK)
		num_rect = num.get_rect(center=(x,y-20))
		self.display.blit(num, num_rect)

		name = self.FNT_SMALL.render("{}".format(self.agents[number][1]), True, self.BLACK)
		name_rect = name.get_rect(center=(x,y+22))
		self.display.blit(name, name_rect)

		if number == self.focus:
			pygame.draw.circle(self.display, self.YELLOW, (x, y), 6)

	def create_agent(self, number):
		name = names[number]
		color = random.choice([0, 1, 2]) # red, green, blue
		shape = random.choice([0, 1, 2]) # square, circle, triangle
		start_x = random.randint(0, (self.display_width//self.grid_size)-1)
		start_y = random.randint(0, (self.display_height//self.grid_size)-1)
		eye = agent.Eye()
		brain = agent.Brain()
		muscle = agent.Muscle()
		brain_map = [[0,0,0],[0,0,0],[0,0,0]]

		self.agents.append([number, name, color, shape, start_x, start_y, eye, brain, muscle, brain_map])

		print("[AGENT#{}] Created! number={}, name={}, color={}, shape={}({}), pos=({}, {})".format(number, number, name, color, shape, self.shapes[shape], start_x, start_y))

	def click_world(self, mouse_x, mouse_y):
		x = mouse_x//self.grid_size
		y = mouse_y//self.grid_size

		agent_pos_list = []
		for agent in self.agents:
			agent_pos_list.append((agent[4], agent[5]))

		for agent, agent_pos in enumerate(agent_pos_list):
			
			if (x, y) == agent_pos and agent != self.focus:
				print("NOW FOLLOWING: #{}, {}".format(agent, self.agents[agent][1]))
				self.focus = agent
				break

	def check_agent(self, number):
		agent = self.agents[number]

		print("[AGENT#{}] Stats:".format(agent[0]))
		print("Name: {}".format(agent[1]))
		print("Color: {}".format(agent[2]))
		print("Shape: {}".format(self.shapes[agent[3]]))
		print("Pos: ({}, {})".format(agent[4], agent[5]))
		print("Experiences: {}".format(agent[7].total_experiences()))

	def display_sidebar(self, x, y):
		basic_info = self.FNT_MEDIUM.render("Steps: {}".format(self.total_steps), True, self.BLACK)
		self.display.blit(basic_info, (x,y))

		# brain maps
		columns = 15
		rows = (len(self.agents) // columns)+1
		size = 36
		num = 0
		for yy in range(rows):
			for xx in range(columns):

				if num <= len(self.agents)-1:
					x1 = x+xx*(size+4)
					y1 = y+yy*(size+16)+48

					# draw number
					name = self.FNT_SMALL.render("#{}".format(num), True, self.BLACK)
					name_rect = name.get_rect(center=(x1+(size/2),y1-8))
					self.display.blit(name, name_rect)

					# draw base
					pygame.draw.rect(self.display, self.GRAY, (x1, y1, size, size))

					# draw contour
					try:
						for sentx in range(3):
							for senty in range(3):
								sent = self.agents[num][7].predict([[sentx, senty]])

								self.agents[num][9][sentx][senty] = sent

								col = self.sentiment_colors[sent]

								pygame.draw.rect(self.display, col, (x1+(sentx*(size/3)), y1+(senty*(size/3)), size/3, size/3))

						# print(self.agents[num][9])

					except sklearn.exceptions.NotFittedError:
						name = self.FNT_SMALL.render("?", True, self.BLACK)
						name_rect = name.get_rect(center=(x1+(size/2),y1+(size/2)))
						self.display.blit(name, name_rect)

					# increment num
					num += 1




Window = Window()

while running:
	Window.main()

	# events
	for event in pygame.event.get():
		if event.type == pygame.MOUSEBUTTONDOWN:
			mouse_pos = pygame.mouse.get_pos()
			Window.click_world(mouse_pos[0], mouse_pos[1])

		if event.type == pygame.QUIT: # quitting
			running = False

pygame.quit()
quit()