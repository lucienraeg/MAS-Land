

agent = [0, 0, 0, 0, 2, 0]

eye_grid = [[0]*7]*7

for xx in range(0, 7):
	for yy in range(0, 7):
		xx1 = xx
		yy1 = yy

		for agent in [agents]:
			print(agent[0])
			if agent[4] == xx1 and agent[5] == yy1: 
				eye_grid[xx][yy] = (xx, yy)

print(eye_grid)