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
		self.display_width = 32*25 # default = 32*40
		self.display_height = 32*25 # default = 32*30
		self.side_display_width = 32*20
		self.side_display_height = 32*0
		self.display = pygame.display.set_mode((self.display_width+self.side_display_width, self.display_height+self.side_display_height))

		# init misc
		self.clock = pygame.time.Clock()
		self.grid_size = 32
		self.total_steps = 1
		self.paused = True
		self.world_speed = 3
		self.world_experience_log_length = 22
		self.world_experience_log = []
		self.world_experience_stats_length = 65
		self.world_experience_stats = [[0]*self.world_experience_log_length]*self.world_experience_stats_length

		# init focus
		self.focus = None # default = None
		self.focus_msg = ("Initiated", 0)

		# init focus graphs
		self.focus_graphs = False
		self.focus_visualize_frequency = 5 # how many steps between
		self.focus_visualize_time = 1 # in seconds

		# init sidebar values
		self.brain_map_frequency = 5 # every n experiences

		# init colors
		self.WHITE = (255, 255, 255)
		self.GRAY = (225, 225, 225)
		self.DK_GRAY = (122, 122, 122)
		self.BLACK = (0, 0, 0)
		self.RED = (255, 0, 0)
		self.GREEN = (0, 255, 0)
		self.BLUE = (0, 0, 255)
		self.YELLOW = (255, 255, 0)

		self.LT_RED = (255, 122, 122)
		self.LT_GREEN = (122, 255, 122)

		# init font
		self.FNT_TINY = pygame.font.SysFont("arial", 9)
		self.FNT_SMALL = pygame.font.SysFont("arial", 11)
		self.FNT_MEDIUM = pygame.font.SysFont("arial", 14)
		self.FNT_LARGE = pygame.font.SysFont("arial", 16)

		# init dicts
		self.colors = {0: self.RED, 1: self.GREEN, 2: self.BLUE}
		self.color_names = {0: "red", 1: "green", 2: "blue"}
		self.shapes = {0: "square", 1: "circle", 2: "triangle"}
		self.sentiments = {-2: "very negative", -1: "negative", 0: "neutral", 1: "positive", 2: "very positive"}
		self.sentiment_colors = {-2: self.RED, -1: self.LT_RED, 0: self.GRAY, 1: self.LT_GREEN, 2: self.GREEN}
		self.sentiment_colors_alt = {-2: self.RED, -1: self.LT_RED, 0: self.BLACK, 1: self.LT_GREEN, 2: self.GREEN}
		self.biomes = {-1: "limbo", 0: "bme0", 1: "bme1", 2: "bme2", 3: "bme3"}
		self.directions = {0: "N", 1: "E", 2: "S", 3: "W"}

		# init agents
		starting_agents = 60 # max = 120
		self.agents = []
		for i in range(starting_agents):
			self.create_agent(i)

	def main(self):
		self.display.fill(self.WHITE)

		self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

		self.draw_grid(self.grid_size)

		if not self.paused:
			agent_pos_list = []
			for agent in self.agents:
				agent_pos_list.append((agent[4], agent[5]))

			for agent in self.agents:
				# if (self.total_steps) % max(1, (60//self.world_speed)) == 0:
				if (self.total_steps+(agent[0]*4)+len(agent[10])) % max(1, (180//self.world_speed)) == 0:
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
							other_biome = -1

							# get features
							X_1 = other_color
							X_2 = other_shape
							X_3 = other_biome

							# decide sentiment (label)
							sent = self.decide_sentiment(agent, X_1, X_2, X_3)
							self.log_experience(self.total_steps, agent[0], a[0], (agent[4], agent[5]), sent)

							# experience
							self.experience(agent, [X_1, X_2, X_3], sent)

							if brain.total_experiences() % self.brain_map_frequency == 0:
								try:
									new_sent = agent[7].predict([[X_1, X_2, X_3]])
									agent[9][X_1][X_2] = new_sent # push to brain map
								except:
									pass

							# print experience
							if agent[0] == self.focus:
								print("[@{}] X = [{} {} {}], y = {}".format(self.total_steps, X_1, X_2, X_3 , sent))
								self.focus_msg = ("#{}, {}, {}, {}: {}".format(other_agent, self.color_names[other_color], self.shapes[other_shape], self.biomes[other_biome], self.sentiments[sent]), sent)

							# learn from experiences
							if brain.total_experiences() > 3:
								brain.learn()

								if agent[0] == self.focus and self.focus_graphs and brain.total_experiences() % self.focus_visualize_frequency == 0:
									# self.check_agent(agent[0])
									brain.visualize("AGENT#{}: {}".format(agent[0], agent[1]), time_limit=self.focus_visualize_time)

					# EYE
					agent_list, agent_pos_list = eye.percieve_area(agent[0], self.agents, agent[4], agent[5])
					agent[11] = agent_pos_list

					# BRAIN
					agent_sent_list = brain.evaluate_agents(self.agents, agent_list)
					agent[12] = agent_sent_list

					# MUSCLE
					sect_scores = muscle.evaluate_directions(agent_pos_list, agent_sent_list)
					sect_best_list = [i for i, j in enumerate(sect_scores) if j == max(sect_scores)]
					chosen_sect = random.choice(sect_best_list)
					agent[13] = sect_scores + [chosen_sect]

					agent[4], agent[5] = muscle.move_direction(agent[4], agent[5], chosen_sect)

					# random movement
					# potential_cells = eye.look(agent[4], agent[5])
					# move_cell = random.choice(potential_cells)
					# agent[4], agent[5] = muscle.move(move_cell[0], move_cell[1])

					# clamp pos
					agent[4] = max(1, min(agent[4], (self.display_width//self.grid_size)-2))
					agent[5] = max(1, min(agent[5], (self.display_height//self.grid_size)-2))

		for agent in self.agents:
			# draw agent
			self.draw_agent(agent[4], agent[5], agent[2], agent[3], agent[0])

		self.display_sidebar(self.display_width+16, 8)
		self.display_controls(self.display_width+16+468-128, 12)

		pygame.display.update()
		self.clock.tick(60)

		if not self.paused:
			self.total_steps += 1

		pygame.display.set_caption("Seed: {}, FPS: {}".format(seed, round(self.clock.get_fps(),2)))

	def log_experience(self, step, agent_a, agent_b, location, sentiment):
		self.world_experience_log = [[step, agent_a, agent_b, location, sentiment]] + self.world_experience_log
		if len(self.world_experience_log) > self.world_experience_log_length:
			self.world_experience_log.pop(self.world_experience_log_length)

		# add to world experience stats
		world_experience_stats_entry = [0]*self.world_experience_log_length
		for i, exp in enumerate(self.world_experience_log):
			world_experience_stats_entry[i] = exp[4]
		self.world_experience_stats = [world_experience_stats_entry] + self.world_experience_stats
		if len(self.world_experience_stats) > self.world_experience_stats_length:
			self.world_experience_stats.pop(self.world_experience_stats_length)

	def experience(self, agent, X, sentiment):
		agent[7].process_experience(X, sentiment)
		agent[10].append(sentiment)

	def decide_sentiment(self, agent, color, shape, biome):
		sent = 0

		self_color = agent[2]
		self_shape = agent[3]

		# if self_color == 0 and color == 1:
		# 	sent = -2
		# elif self_color == 1 and color == 2:
		# 	sent = -2
		# elif self_color == 2 and color == 0:
		# 	sent = -2
		# else:
		# 	if self_shape == shape:
		# 		sent = 2

		if self_color != color:
			sent = -2
		else:
			sent = 2

		# sent = random.randint(-2, 2)

		return sent

	def draw_grid(self, grid_size):
		for col in range((self.display_width//grid_size)+1):
			pygame.draw.line(self.display, self.GRAY, (col*grid_size, 0), (col*grid_size, self.display_height))

		for row in range((self.display_height//grid_size)+1):
			pygame.draw.line(self.display, self.GRAY, (0, row*grid_size), (self.display_width, row*grid_size))

	def draw_agent(self, x, y, color, shape, number):
		x = x*self.grid_size+(self.grid_size//2)
		y = y*self.grid_size+(self.grid_size//2)

		self.draw_agent_body(x, y, color, shape)

		num = self.FNT_SMALL.render("#{}".format(number), True, self.BLACK)
		num_rect = num.get_rect(center=(x,y-20))
		self.display.blit(num, num_rect)

		name = self.FNT_SMALL.render("{}".format(self.agents[number][1]), True, self.BLACK)
		name_rect = name.get_rect(center=(x,y+22))
		self.display.blit(name, name_rect)

		if number == self.focus:
			pygame.draw.rect(self.display, self.YELLOW, (x-32, y-32, 64, 64), 3)

	def draw_agent_body(self, x, y, color, shape):
		if self.shapes[shape] == "square":
			pygame.draw.rect(self.display, self.colors[color], (x-10, y-10, 20, 20))
			pygame.draw.rect(self.display, self.BLACK, (x-11, y-11, 22, 22), 3)
		elif self.shapes[shape] == "circle":
			pygame.draw.circle(self.display, self.colors[color], (x, y), 12)
			pygame.draw.circle(self.display, self.BLACK, (x, y), 13, 3)
		elif self.shapes[shape] == "triangle":
			pygame.draw.polygon(self.display, self.colors[color], ((x, y-10), (x-10, y+10), (x+10, y+10)))
			pygame.draw.polygon(self.display, self.BLACK, ((x, y-12), (x-12, y+12), (x+12, y+12)), 3)

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
		experience_history = []
		eye_data = []
		brain_data = []
		muscle_data = []

		self.agents.append([number, name, color, shape, start_x, start_y, eye, brain, muscle, brain_map, experience_history, eye_data, brain_data, muscle_data])

		print("[AGENT#{}] Created! number={}, name={}, color={}({}), shape={}({}), pos=({}, {})".format(number, number, name, color, self.color_names[color], shape, self.shapes[shape], start_x, start_y))

	def display_controls(self, x, y):
		
		# auto on/off
		pygame.draw.rect(self.display, self.GRAY, (x-76, y, 64, 16))

		if x-76 < self.mouse_x < x-16 and y < self.mouse_y < y+16:
			pygame.draw.rect(self.display, self.BLACK, (x-76, y, 64, 16), 2)
			if pygame.mouse.get_pressed()[0]:
				self.paused = 1

		text = self.FNT_SMALL.render("PAUSE", True, self.BLACK)
		text_rect = text.get_rect(center=(x-76+32,y+8))
		self.display.blit(text, text_rect)

		if self.paused: col = self.LT_GREEN
		else: col = self.GRAY

		pygame.draw.rect(self.display, col, (x-76-68, y, 64, 16))

		if x-76-68 < self.mouse_x < x-16-68 and y < self.mouse_y < y+16:
			pygame.draw.rect(self.display, self.BLACK, (x-76-68, y, 64, 16), 2)
			if pygame.mouse.get_pressed()[0]:
				self.paused = 0

		text = self.FNT_SMALL.render("PLAY", True, self.BLACK)
		text_rect = text.get_rect(center=(x-76+32-68,y+8))
		self.display.blit(text, text_rect)

		# bar
		sensitivity = 0.25

		w = round(self.world_speed,1)
		success = min(1.0, self.clock.get_fps() / 60)

		if success < 0.6: col = self.RED
		elif success < 0.8: col = self.LT_RED
		else: col = self.DK_GRAY

		pygame.draw.rect(self.display, self.GRAY, (x, y, 256, 16))
		pygame.draw.rect(self.display, col, (x, y, self.world_speed/sensitivity, 16))

		if x < self.mouse_x < x+256 and y < self.mouse_y < y+16:
			pygame.draw.rect(self.display, self.BLACK, (x, y, 256, 16), 2)

			if pygame.mouse.get_pressed()[0]:
				xx = self.mouse_x-x
				self.world_speed = max(1, xx*sensitivity)

		text = self.FNT_SMALL.render("{} [{}%]".format(w, round(success*100, 1)), True, self.BLACK)
		text_rect = text.get_rect(center=(x+128,y+8))
		self.display.blit(text, text_rect)

	def display_sidebar(self, x, y):
		basic_info = self.FNT_MEDIUM.render("Steps: {}".format(self.total_steps), True, self.BLACK)
		self.display.blit(basic_info, (x,y+3))

		# brain maps
		section_1_y = 48
		columns = 15
		rows = ((len(self.agents)-1) // columns)+1
		size = 36
		num = 0
		for yy in range(rows):
			for xx in range(columns):
				if num <= len(self.agents)-1:
					x1 = x+xx*(size+4)
					y1 = y+yy*(size+16)+section_1_y

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
								col = self.sentiment_colors[self.agents[num][9][sentx][senty]]
								pygame.draw.rect(self.display, col, (x1+(sentx*(size/3)), y1+(senty*(size/3)), size/3, size/3))
 
					except sklearn.exceptions.NotFittedError:
						pass

					# draw highlight box				
					if x1 < self.mouse_x < x1+size and y1 < self.mouse_y < y1+size and self.focus != num:
						pygame.draw.rect(self.display, self.BLACK, (x1, y1, size, size), 1)
						# self.draw_agent_body(x1+(size//2), y1+(size//2), self.agents[num][2], self.agents[num][3])

						if pygame.mouse.get_pressed()[0]:
							self.focus_msg = ("new focus", 0)
							self.focus = num

					if self.focus == num:
						pygame.draw.rect(self.display, self.BLACK, (x1, y1, size, size), 3)

					# increment num
					num += 1

		section_2_y = section_1_y + (rows*(size+16)) + 16
					
		# world stats
		if self.focus == None:
			column_names = ["Step #", "Agent A", "Agent B", "Location", "Sentiment"]

			text = self.FNT_MEDIUM.render("World Experience Log", True, self.BLACK)
			self.display.blit(text, (x, section_2_y))

			pygame.draw.rect(self.display, self.BLACK, (x, section_2_y+20, 310, 5+(self.world_experience_log_length)*18), 2)

			for row in range(0, self.world_experience_log_length):

				if row != 0:
					try:
						if self.world_experience_log[row][4] == 0: bgcol = self.GRAY
						elif self.world_experience_log[row][4] > 0: bgcol = self.LT_GREEN
						elif self.world_experience_log[row][4] < 0: bgcol = self.LT_RED
					except:
						bgcol = self.GRAY

					pygame.draw.rect(self.display, bgcol, (x+5, section_2_y+(row*18)+22, 300, 16))

				for column in range(0, 5):
					x1 = x+(column*60)+5
					y1 = section_2_y+(row*18)+22

					if row == 0:
						t = column_names[column]
					else:
						try:
							t = self.world_experience_log[row][column]
						except:
							t = ""

					text = self.FNT_MEDIUM.render("{}".format(t), True, self.BLACK)
					self.display.blit(text, (x1, y1))


			# recent experience map
			text = self.FNT_MEDIUM.render("Recent Experience Map", True, self.BLACK)
			self.display.blit(text, (x+320, section_2_y))

			pygame.draw.rect(self.display, self.BLACK, (x+320, section_2_y+20, 275, 275), 2)

			for exp in self.world_experience_log:
				world_diff_x = 275/self.display_width
				world_diff_y = 275/self.display_height
				exp_x = x+328+(exp[3][0]*(32*world_diff_x))
				exp_y = section_2_y+28+(exp[3][1]*(32*world_diff_y))
				pygame.draw.circle(self.display, self.sentiment_colors_alt[exp[4]], (int(exp_x), int(exp_y)), 4)

			# world experience stats
			text = self.FNT_MEDIUM.render("World Experience Stats", True, self.BLACK)
			self.display.blit(text, (x+320, section_2_y+300))

			pygame.draw.rect(self.display, self.BLACK, (x+320, section_2_y+300+20, 275, 101), 2)

			for i, exp_set in enumerate(self.world_experience_stats):
				exp_list = []
				for j, exp in enumerate(sorted(exp_set, reverse=True)):
					color = self.sentiment_colors[exp]
					pygame.draw.rect(self.display, color, (x+320+8+(i*4), section_2_y+300+20+7+(j*4), 3, 4))



		# agent focus
		if self.focus != None:
			a = self.agents[self.focus]

			agent_title = self.FNT_LARGE.render("AGENT#{}: {}".format(a[0], a[1]), True, self.BLACK, self.GRAY)
			self.display.blit(agent_title, (x,y+section_2_y))

			# agent info text
			agent_color = self.FNT_MEDIUM.render("Color: {} ({})".format(self.color_names[a[2]].title(), a[2]), True, self.BLACK)
			agent_shape = self.FNT_MEDIUM.render("Shape: {} ({})".format(self.shapes[a[3]], a[3]).title(), True, self.BLACK)
			agent_pos = self.FNT_MEDIUM.render("Pos: ({}, {})".format(a[4], a[5]).title(), True, self.BLACK)

			self.display.blit(agent_color, (x+32,y+section_2_y+20))
			self.display.blit(agent_shape, (x+32,y+section_2_y+20+16))
			self.display.blit(agent_pos, (x,y+section_2_y+20+32))

			self.draw_agent_body(x+16, y+section_2_y+20+16, a[2], a[3])

			# agent experience history
			exp_size = 4
			exp_shown_length = 368//exp_size

			for i, exp in enumerate(a[10][-exp_shown_length:]):
				pygame.draw.rect(self.display, self.sentiment_colors[exp], ((x+220)+i*exp_size, y+section_2_y+20, exp_size, 30))

			text = self.FNT_MEDIUM.render("Last Experience:", True, self.DK_GRAY)
			self.display.blit(text, (x+220,y+section_2_y))

			agent_experiences = self.FNT_MEDIUM.render("Total Experiences: {}".format(a[7].total_experiences()), True, self.BLACK)
			self.display.blit(agent_experiences, (x+220,y+section_2_y+20+32))

			agent_focus_msg = self.FNT_MEDIUM.render("{}".format(self.focus_msg[0]), True, self.sentiment_colors_alt[self.focus_msg[1]])
			agent_focus_msg_rect = agent_focus_msg.get_rect()
			agent_focus_msg_rect.right = x+220+368
			agent_focus_msg_rect.top = y+section_2_y
			self.display.blit(agent_focus_msg, agent_focus_msg_rect)

			agent_experience_rate = self.FNT_MEDIUM.render("{} per ksteps".format(round(a[7].total_experiences()/self.total_steps*1000)), True, self.BLACK)
			agent_experience_rate_rect = agent_experience_rate.get_rect()
			agent_experience_rate_rect.right = x+220+368
			agent_experience_rate_rect.top = y+section_2_y+20+32
			self.display.blit(agent_experience_rate, agent_experience_rate_rect)

		# agent sections
		section_3_y = section_2_y + 128
		if self.focus != None:
			a = self.agents[self.focus]

			# eye
			x1, y1 = x, section_3_y

			text = self.FNT_LARGE.render("Eye", True, self.BLACK)
			self.display.blit(text, (x1, y1-24))
			pygame.draw.rect(self.display, self.BLACK, (x1, y1, 160, 160), 2)

			# eye grid
			for yy in range(7):
				for xx in range(7):
					if (xx, yy) == (3, 3):
						col = self.BLACK
					elif (xx, yy) in a[11]:
						col = self.DK_GRAY
					else:
						col = self.GRAY

					pygame.draw.rect(self.display, col, (x1+(xx*20)+11, y1+(yy*20)+11, 19, 19))

			text = self.FNT_LARGE.render("{}".format(len(a[11])), True, self.BLACK)
			self.display.blit(text, (x1+4, y1+1))

			# text below
			text = self.FNT_SMALL.render("{}".format(a[11][:5]), True, self.BLACK)
			self.display.blit(text, (x1, y1+162))

			# eye to brain link
			pygame.draw.line(self.display, self.BLACK, (x1+160, y1+80), (x1+220, y1+80), 2)

			# brain
			x1, y1 = x+220, section_3_y

			text = self.FNT_LARGE.render("Brain", True, self.BLACK)
			self.display.blit(text, (x1, y1-24))
			pygame.draw.rect(self.display, self.BLACK, (x1, y1, 160, 160), 2)

			# brain grid
			for yy in range(7):
				for xx in range(7):
					if (xx, yy) in a[11]:
						try:
							index = a[11].index((xx, yy))
							sent = a[12][index]
							col = self.sentiment_colors_alt[sent]
						except:
							col = self.DK_GRAY
					else:
						col = self.GRAY

					pygame.draw.rect(self.display, col, (x1+(xx*20)+11, y1+(yy*20)+11, 19, 19))

			pygame.draw.rect(self.display, self.BLACK, (x1+71, y1+71, 19, 19), 1)

			# text below
			text = self.FNT_SMALL.render("{}".format(a[12][:9]), True, self.BLACK)
			self.display.blit(text, (x1, y1+162))

			# brain to muscle link
			pygame.draw.line(self.display, self.BLACK, (x1+160, y1+80), (x1+220, y1+80), 2)

			# muscle
			x1, y1 = x+440, section_3_y

			text = self.FNT_LARGE.render("Muscle", True, self.BLACK)
			self.display.blit(text, (x1, y1-24))
			pygame.draw.rect(self.display, self.BLACK, (x1, y1, 160, 160), 2)

			# visualization of muscle preference
			pygame.draw.rect(self.display, self.BLACK, (x1+80-8, y1+80-8, 16, 16))

			sect_scores = a[13]

			# draw lines for each direction (NESW)
			if len(sect_scores) > 0:
				if sect_scores[0] != 0:
					if sect_scores[0] >= 0: col = self.GREEN
					else: col = self.RED
					pygame.draw.line(self.display, col, (x1+80, y1+80-8), (x1+80, y1+80-8-min(64, abs((sect_scores[0]+1)*8))), 9)

				if sect_scores[1] != 0:
					if sect_scores[1] >= 0: col = self.GREEN
					else: col = self.RED
					pygame.draw.line(self.display, col, (x1+80+8, y1+80), (x1+80+8+min(64, abs((sect_scores[1]+1)*8)), y1+80), 9)

				if sect_scores[2] != 0:
					if sect_scores[2] >= 0: col = self.GREEN
					else: col = self.RED
					pygame.draw.line(self.display, col, (x1+80, y1+80+8), (x1+80, y1+80+8+min(64, abs((sect_scores[2]+1)*8))), 9)

				if sect_scores[3] != 0:
					if sect_scores[3] >= 0: col = self.GREEN
					else: col = self.RED
					pygame.draw.line(self.display, col, (x1+80-8, y1+80), (x1+80-8-min(64, abs((sect_scores[3]+1)*8)), y1+80), 9)

				# text for chosen direction
				text = self.FNT_LARGE.render("{}".format(self.directions[sect_scores[4]]), True, self.BLACK)
				self.display.blit(text, (x1+4, y1+4))

				if sect_scores[4] == 0: pygame.draw.line(self.display, self.BLACK, (x1+80, y1+80), (x1+80, y1+80-60), 3)
				elif sect_scores[4] == 1: pygame.draw.line(self.display, self.BLACK, (x1+80, y1+80), (x1+80+60, y1+80), 3)
				elif sect_scores[4] == 2: pygame.draw.line(self.display, self.BLACK, (x1+80, y1+80), (x1+80, y1+80+60), 3)
				elif sect_scores[4] == 3: pygame.draw.line(self.display, self.BLACK, (x1+80, y1+80), (x1+80-60, y1+80), 3)

				# text below, direction scores
				text = self.FNT_SMALL.render("{}".format(sect_scores[:4]), True, self.BLACK)
				self.display.blit(text, (x1, y1+162))



Window = Window()

while running:
	Window.main()

	mouse_pos = pygame.mouse.get_pos()

	if pygame.mouse.get_pressed()[0] and mouse_pos[0] < Window.display_width:
		Window.focus = None

	# events
	for event in pygame.event.get():
		if event.type == pygame.QUIT: # quitting
			running = False

pygame.quit()
quit()