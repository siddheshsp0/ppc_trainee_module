import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np


def main():
    # Importing dataset
    waypoints = pd.read_csv('loop_track_waypoints.csv')
    waypoints = waypoints.drop('Index', axis=1)
    waypoints = waypoints.to_numpy()
    sum=0

    # Initialising parameter t
    t = np.zeros((waypoints.shape[0],))
    for i in range(waypoints.shape[0]-1):
        t[i] = sum
        sum+=np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2+(waypoints[i+1][1] - waypoints[i][1])**2)
    t[-1] = sum

    # Spline over x
    spline_x = CubicSpline(t, waypoints[:, 0])
    spline_y = CubicSpline(t, waypoints[:,1])
    t_continous = np.linspace(t[0], t[-1], 1000)
    x_continous = spline_x(t_continous)
    y_continous = spline_y(t_continous)

    plt.scatter(waypoints[:,0].tolist(), waypoints[:,1].tolist(), label="Waypoints", color="green")
    plt.plot(x_continous, y_continous, label="Cubic Spline using scipy")
    plt.legend()
    plt.grid(True)
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.show()


if __name__=='__main__':
    main()