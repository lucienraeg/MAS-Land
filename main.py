import pygame
import random

seed = 12345678
running = True

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

	def main(self):
		self.display.fill(self.WHITE)

		self.draw_grid(self.grid_size)
		self.draw_agent(2, 3, self.RED, "square")
		self.draw_agent(4, 2, self.GREEN, "circle")
		self.draw_agent(4, 5, self.BLUE, "triangle")

		pygame.display.update()
		self.clock.tick(15)

		pygame.display.set_caption("Seed: {}, FPS: {}".format(seed, round(self.clock.get_fps(),2)))

	def draw_grid(self, grid_size):
		for col in range((self.display_width//grid_size)+1):
			pygame.draw.line(self.display, self.GRAY, (col*grid_size, 0), (col*grid_size, self.display_height))

		for row in range((self.display_height//grid_size)+1):
			pygame.draw.line(self.display, self.GRAY, (0, row*grid_size), (self.display_width, row*grid_size))

	def draw_agent(self, row, col, color, shape):
		x = row*self.grid_size+(self.grid_size//2)
		y = col*self.grid_size+(self.grid_size//2)

		if shape == "square":
			pygame.draw.rect(self.display, color, (x-10, y-10, 20, 20))
			pygame.draw.rect(self.display, self.BLACK, (x-11, y-11, 22, 22), 3)
		elif shape == "circle":
			pygame.draw.circle(self.display, color, (x, y), 12)
			pygame.draw.circle(self.display, self.BLACK, (x, y), 13, 3)
		elif shape == "triangle":
			pygame.draw.polygon(self.display, color, ((x, y-10), (x-10, y+10), (x+10, y+10)))
			pygame.draw.polygon(self.display, self.BLACK, ((x, y-11), (x-11, y+11), (x+11, y+11)), 3)






Window = Window()

while running:
	Window.main()

	# allow quitting
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

pygame.quit()
quit()