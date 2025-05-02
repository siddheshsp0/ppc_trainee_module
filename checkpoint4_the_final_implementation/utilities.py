import numpy as np
def stanley_steering(x, y, yaw, v, waypoints, k=1.0, ks=1e-2, max_steer=np.radians(30)):
    """
    Stanley steering controller.

    Args:
        x, y     : rear axle position of the car
        yaw      : vehicle heading angle (in radians)
        v        : vehicle speed
        waypoints: Nx2 array of path waypoints
        k        : cross-track gain
        ks       : softening term to prevent div by zero
        max_steer: steering angle limits (in radians)

    Returns:
        steer       : steering angle in radians
        target_idx  : index of the nearest waypoint
    """
    # Step 1: Compute front axle position
    L = 2.5  # assume fixed wheelbase
    fx = x + L * np.cos(yaw)
    fy = y + L * np.sin(yaw)

    # Step 2: Find nearest waypoint
    dists = np.array([np.sqrt((fx-point[0])**2 + (fy-point[1])**2) for point in waypoints])
    target_idx = np.argmin(dists)
    closest_dist = dists[target_idx]



    # Step 3: Compute heading of path at that point
    if target_idx != 0:
        dx = waypoints[target_idx][0] - waypoints[target_idx - 1][0]
        dy = waypoints[target_idx][1] - waypoints[target_idx - 1][1]
        heading = np.arctan2(dy, dx)
    else:
        dx = waypoints[1][0] - waypoints[0][0]
        dy = waypoints[1][1] - waypoints[0][1]
        heading = np.arctan2(dy, dx)


    # Step 4: Compute heading error
    heading_error = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
    # heading_error = heading - yaw


    # Step 5: Compute signed cross-track error
    # This code was chatgpted
    dx = waypoints[target_idx][0] - fx
    dy = waypoints[target_idx][1] - fy
    path_heading = heading  # already computed above
    cross = np.sin(path_heading) * dx - np.cos(path_heading) * dy
    cte = np.sign(cross) * closest_dist

    



    # Step 6: Compute steering using Stanley law
    steer = heading_error + np.arctan(k*cte/(v+ks))# write your code here
    steer = np.clip(steer, -max_steer, max_steer) # clip to max_steer

    return steer, target_idx


class PIDController:
    def __init__(self, Kp, Ki, Kd,):
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
        self.integral_history=0
        self.error_history=0
    def compute_throttle(self, error, dt):
        self.integral_history+=error*dt
        throttle = self.Kp*error + self.Ki*self.integral_history + self.Kd*((error-self.error_history)/dt)
        self.error_history=error
        return throttle