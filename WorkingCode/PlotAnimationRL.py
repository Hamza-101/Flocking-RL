import json
from Params import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from tqdm import tqdm
import os
fig, ax = plt.subplots()

def read_data(file):
    with open(file, "r") as f:
        all_data = json.load(f)
    agent_data = {}

    for agent_id, agent_points in all_data.items():
        agent_data[int(agent_id)] = agent_points

    max_num_points = max(len(agent_points) for agent_points in agent_data.values())

    return agent_data, max_num_points

def function(i, agent_data):
    plt.clf()
    plt.cla()

    plt.axis('equal')  

    for agent_id, data_points in agent_data.items():
        if i < len(data_points):
            data_point = data_points[i]
            x_pos = data_point[0] 
            y_pos = data_point[1] 

            plt.scatter(x_pos, y_pos, c='r', marker='o', s=20)

            circle = Circle((x_pos, y_pos), radius=5, edgecolor='black', facecolor='none', linewidth=0.1)
            plt.gca().add_patch(circle)

    plt.grid(True)

num_timesteps = 3000  

file=Results["Positions"] 
agent_data, max_num_points = read_data(file)

    #Make directory
name, _ =os.path.splitext(file)
print(name)

for i in tqdm(range(Animation["TimeSteps"])):
    function(i, agent_data)

anim = animation.FuncAnimation(fig, function, fargs=(agent_data,), frames=Animation["TimeSteps"], interval=100, blit=False)
anim.save(rf"{name}.mp4", writer="ffmpeg")
