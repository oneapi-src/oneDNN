#! /bin/python3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time

class data_points:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.zs = []

    def add(self, x, y, z):
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)

plots = {};

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.draw()
plt.pause(0.2)

x_label = None
y_label = None
z_label = None

def update(last_update):
    current = time.monotonic()
    if(current - last_update > 1): # Occasionally update data
        start = current
        ax.cla()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        for key, value in plots.items():
            ax.scatter(value.xs, value.ys, value.zs, s=0.5, label=key)
        ax.legend()
        plt.draw()
        last_update = current
    plt.pause(0.02)
    return last_update

last_update = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break;
    if not line.startswith("plot"):
        continue
    print(line, end="")

    if(x_label == None):
        _, x_label, y_label, z_label, _ = line.strip().split(", ")
        continue

    _, k, n, bandwidth, plot_id = line.strip().split(", ")
    if not plot_id in plots:
        plots[plot_id] = data_points()
    plots[plot_id].add(int(k), int(n), float(bandwidth))
    last_update = update(last_update)

plt.ioff()
plt.show()
