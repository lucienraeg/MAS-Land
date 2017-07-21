import pygame
import random
import csv
import agent

names = []

with open("agent-names.csv", "rt") as f:
	reader = csv.reader(f)
	for row in reader:
		names.append(row[0])

seed = 12341236
running = True

random.seed(seed)

class Window:

	def __init__(self):
		pygame.init()

		# init display
		self.display_width = 32*30
		self.display_height = 32*20
		self.display = pygame.display.set_mode((self.display_width, self.display_height))

		# init misc
		self.clock = pygame.time.Clock()
		self.grid_size = 32

		# init colors
		self.WHITE = (255, 255, 255)
		self.GRAY = (225, 225, 225)
		self.BLACK = (0, 0, 0)
		self.RED = (255, 0, 0)
		self.GREEN = (0, 255, 0)
		self.BLUE = (0, 0, 255)

		# init font
		self.FNT_SMALL = pygame.font.SysFont("arial", 11)

		# init dicts
		self.colors = {0: self.RED, 1: self.GREEN, 2: self.BLUE}
		self.shapes = {0: "square", 1: "circle", 2: "triangle"}

		# init agents
		starting_agents = 4
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

			# process any experience had
			agent_contacts = []
			for a in enumerate(agent_pos_list):
				# check for contact with another agent
				if a[1] == (agent[4], agent[5])  and a[0] != agent[0]:
					print("[AGENT#{}] Contact! pos=({},{})".format(agent[0], a[1][0], a[1][1]))
					

			eye = agent[6]
			brain = agent[7]
			muscle = agent[8]

			# percieve area
			potential_cells = eye.look(agent[4], agent[5])

			# calculate best cell to move to
			move_cell = random.choice(potential_cells)

			# move to that cell
			agent[4], agent[5] = muscle.move(move_cell[0], move_cell[1])

			# clamp pos
			agent[4] = max(1, min(agent[4], (self.display_width//self.grid_size)-2))
			agent[5] = max(1, min(agent[5], (self.display_width//self.grid_size)-2))

			# draw agent
			self.draw_agent(agent[4], agent[5], agent[2], agent[3], agent[0])

		pygame.display.update()
		self.clock.tick(2)

		pygame.display.set_caption("Seed: {}, FPS: {}".format(seed, round(self.clock.get_fps(),2)))

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


	def create_agent(self, number):
		name = names[number]
		color = random.choice([0, 1, 2]) # red, green, blue
		shape = random.choice([0, 1, 2]) # square, circle, triangle
		start_x = random.randint(0, (self.display_width//self.grid_size)-1)
		start_y = random.randint(0, (self.display_height//self.grid_size)-1)
		eye = agent.Eye()
		brain = agent.Brain()
		muscle = agent.Muscle()

		self.agents.append([number, name, color, shape, start_x, start_y, eye, brain, muscle])

		print("Agent Created: number={}, name={}, color={}, shape={}({}), pos=({}, {})".format(number, name, color, shape, self.shapes[shape], start_x, start_y))


Window = Window()

while running:
	Window.main()

	# allow quitting
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

pygame.quit()
quit()