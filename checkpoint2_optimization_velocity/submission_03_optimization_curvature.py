# ---------------- Importing code from checkpoint 1 for the function interpolate ------------------------ #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_coeff(coeff, iterator, eqn_no, x, negative=False):
    factor = -1 if negative else 1
    coeff[0][iterator][eqn_no*4] = (factor)*(x**3)
    coeff[0][iterator][eqn_no*4+1] = (factor)*(x**2)
    coeff[0][iterator][eqn_no*4+2] = (factor)*x
    coeff[0][iterator][eqn_no*4+3] = (factor)*1.0
    return coeff
def set_coeff_first_derivative(coeff, iterator, eqn_no, x, negative=False):
    factor = -1 if negative else 1
    coeff[0][iterator][eqn_no*4] = (factor)*3.0*(x**2)
    coeff[0][iterator][eqn_no*4+1] = (factor)*2.0*x
    coeff[0][iterator][eqn_no*4+2] = (factor)*1.0
    coeff[0][iterator][eqn_no*4+3] = (factor)*0.0
    return coeff
def set_coeff_second_derivative(coeff, iterator, eqn_no, x, negative=False):
    factor = -1 if negative else 1
    coeff[0][iterator][eqn_no*4] = (factor)*6.0*x
    coeff[0][iterator][eqn_no*4+1] = (factor)*2.0
    coeff[0][iterator][eqn_no*4+2] = (factor)*0.0
    coeff[0][iterator][eqn_no*4+3] = (factor)*0.0
    return coeff
    


def interpolate(x, t, base_1=0.0, base_n=0.0):
    '''
    Takes x and t, and interpolates x w.r.t t, and returns an array of coefficients [a,b,c,d]. The equations are solved using matrix multiplication.
    Return type is a numpy array where coefficients of equation i are [a,b,c,d]=[coefficients[4*i], coefficients[4*i+1], coefficients[4*i+2], coefficients[4*i+3], ]
    '''

    # Iterator to keep track of the index of the equation
    iterator=0
    # Checking dimensions of x and t
    if (x.shape[0] != t.shape[0]):
        raise Exception("Dimension of x and t passed are not equal")
    num_eqn = x.shape[0]-1
    # Initialising matrices
    coefficients = np.zeros(((num_eqn)*4,)) # ak, bk, ck, dk, will be at index 4*(k-1), 4*(k-1)+1, 4*(k-1)+2, 4*(k-1)+3
    B_matrix = np.zeros(((num_eqn)*4,)) # Initialising 0 matrix at the right side of the equation
    calc_matrix = [np.zeros(((num_eqn)*4, (num_eqn)*4))] # This will contain num_eqn*4 equations, along with num_eqn*4 coefficients


    # Boundary Conditions (2nd order)
    set_coeff_second_derivative(calc_matrix, iterator, 0, t[0])
    B_matrix[iterator] = base_1
    iterator+=1
    set_coeff_second_derivative(calc_matrix, iterator, -1, t[-1])
    B_matrix[iterator] = base_n
    iterator+=1

    for i in range(num_eqn-1):
    # Using coordinates (0th order) and 
        set_coeff(calc_matrix, iterator, i, t[i],)
        B_matrix[iterator] = x[i]
        iterator+=1
        set_coeff(calc_matrix, iterator, i, t[i+1],)
        B_matrix[iterator] = x[i+1]
        iterator+=1

    # Equating slopes between two adjacent waypoints (1st order)
        set_coeff_first_derivative(calc_matrix, iterator, i, t[i+1])
        set_coeff_first_derivative(calc_matrix, iterator, i+1, t[i+1], negative=True)
        iterator+=1
    # Equating slopes of slopes between two adjacent waypoints (2nd order)
        set_coeff_second_derivative(calc_matrix, iterator, i, t[i+1])
        set_coeff_second_derivative(calc_matrix, iterator, i+1, t[i+1], negative=True)
        iterator+=1

    # These 2 eqns were not included in above loop to avoid the 1st and 2nd order equation part going out of bound
    set_coeff(calc_matrix, iterator, -1, t[-2],)
    B_matrix[iterator] = x[-2]
    iterator+=1
    set_coeff(calc_matrix, iterator, -1, t[-1],)
    B_matrix[iterator] = x[-1]
    iterator+=1

    coefficients = np.linalg.solve(calc_matrix[0], B_matrix)
    return coefficients



# ---------------------- Checkpoint 1 code end ----------------- #
from scipy.optimize import minimize



def calc_curvature(x_t, y_t, t):
    '''
    Function that calculates curvature at point (x,y), parametrized by the variable t.
    x_t: numpy array, shape=(4,), contains [a,b,c,d], for f(t)=x, indexed.
    y_t: numpy array, shape=(4,), contains [a,b,c,d], for g(t)=y, indexed.
    a,b,c,d correspond to u(t)=at^3+bt^2+ct+d
    '''
    # Calculating first and second derivatives
    fd_x=3*x_t[0]*(t**2)+ 2*x_t[1]*(t**1)+x_t[2]
    fd_y=3*y_t[0]*(t**2)+ 2*y_t[1]*(t**1)+y_t[2]
    sd_x=6*x_t[0]*t+2*x_t[1]
    sd_y=6*y_t[0]*t+2*y_t[1]
    # Returns calculated curvature
    return (np.abs(fd_x*sd_y - fd_y*sd_x))/((fd_x**2 + fd_y**2)**(3/2))

def integrate(func, x_t, y_t, n_points, t_low, t_high):
    '''
    Performs numerical integration on the function func from t_low to t_high, with step size (t_low-t_high)/(n_points-1)
    x_t and y_t are parameters to func
    '''
    pts = np.linspace(t_low, t_high, n_points)[:-1]
    val=0
    for i in pts:
        val+=func(x_t, y_t, i)*(t_high-t_low)/(n_points-1)
    return val



def main():
    # Importing dataset and defining defaults
    base_1_default = 10.0 # Set base values away from 0.0 (absurd) to see the difference between default and optimised in graphs
    base_n_default = 10.0
    waypoints = pd.read_csv('../checkpoint1_interpolation/loop_track_waypoints.csv')
    waypoints = waypoints.drop('Index', axis=1)
    waypoints = waypoints.to_numpy()[:10] # Included only 10 waypoints to see the difference bettween optimised and unoptimised paths
    t = np.zeros((waypoints.shape[0],))
    # Calculating parameter t (parametrized euclidean distance wise)
    sum=0
    for i in range(waypoints.shape[0]-1):
        t[i] = sum
        sum+=np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2+(waypoints[i+1][1] - waypoints[i][1])**2)
    t[-1] = sum

    # Defining a loss function
    def loss(inpt):
        base_1, base_n = inpt

        # Interpolate with given conditions
        coeff_x = interpolate(waypoints[:,0], t, base_1, base_n)
        coeff_y = interpolate(waypoints[:,1], t, base_1, base_n)

        loss=0.0
        for i in range(t.shape[0]-1):
            loss+=integrate(calc_curvature, coeff_x[4*i:4*i+4], coeff_y[4*i:4*i+4], 100, t[i], t[i+1]) # Increase parameter number 4 for higher number of points for integration
        return loss
    

    # Applying regression (Used a library)
    base_1_optimal, base_n_optimal = minimize(loss, x0=[base_1_default, base_n_default], tol=1e-6).x # tol is tolerance (Threshold)
    print("Calculated Optimal double derivative at the 1st point is: "+str(base_1_optimal))
    print("Calculated Optimal double derivative at the last point is: "+str(base_n_optimal))

    # Calculating optimal coefficients and not very optimal coefficients
    coeff_x_optimised = interpolate(waypoints[:,0], t, base_1_optimal, base_n_optimal)
    coeff_y_optimised = interpolate(waypoints[:,1], t, base_1_optimal, base_n_optimal)
    coeff_x_default = interpolate(waypoints[:,0], t, base_1_default, base_n_default)
    coeff_y_default = interpolate(waypoints[:,1], t, base_1_default, base_n_default)

    # Plotting the final optimal path vs unoptimised (base cases 0.0) path(Copied from my submission for checkpoint 1)
    x_vals_optimised = []
    y_vals_optimised = []
    x_vals_def = []
    y_vals_def = []
    t_vals = []
    for i in range(waypoints.shape[0]-1):
        a_opt = [coeff_x_optimised[i*4], coeff_y_optimised[i*4]]
        b_opt = [coeff_x_optimised[i*4+1], coeff_y_optimised[i*4+1]]
        c_opt = [coeff_x_optimised[i*4+2], coeff_y_optimised[i*4+2]]
        d_opt = [coeff_x_optimised[i*4+3], coeff_y_optimised[i*4+3]]
        a_def = [coeff_x_default[i*4], coeff_y_default[i*4]]
        b_def = [coeff_x_default[i*4+1], coeff_y_default[i*4+1]]
        c_def = [coeff_x_default[i*4+2], coeff_y_default[i*4+2]]
        d_def = [coeff_x_default[i*4+3], coeff_y_default[i*4+3]]
        
        t_continuous = np.linspace(t[i], t[i+1], 100)
        x_continuous_opt = a_opt[0] * t_continuous**3 + b_opt[0] * t_continuous**2 + c_opt[0] * t_continuous + d_opt[0]
        y_continuous_opt = a_opt[1] * t_continuous**3 + b_opt[1] * t_continuous**2 + c_opt[1] * t_continuous + d_opt[1]
        x_continuous_def = a_def[0] * t_continuous**3 + b_def[0] * t_continuous**2 + c_def[0] * t_continuous + d_def[0]
        y_continuous_def = a_def[1] * t_continuous**3 + b_def[1] * t_continuous**2 + c_def[1] * t_continuous + d_def[1]
        
        x_vals_optimised.extend(x_continuous_opt)
        y_vals_optimised.extend(y_continuous_opt)
        x_vals_def.extend(x_continuous_def)
        y_vals_def.extend(y_continuous_def)
        t_vals.extend(t_continuous)
    
    plt.plot(x_vals_optimised, y_vals_optimised, label="Optimised Path", color="blue", zorder=3)
    plt.plot(x_vals_def, y_vals_def, label="Default Unoptimised Path", color="red", zorder=2)

    # Plot the original waypoints (from csv filee)
    plt.scatter(waypoints[:,0], waypoints[:,1], color="green", label="Waypoints", zorder=5)
    plt.xlabel("x Coordinate")
    plt.ylabel("y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()




# Entry point

if __name__ == '__main__':
    main()