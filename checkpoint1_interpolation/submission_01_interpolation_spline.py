# Application of cubic spline between n points
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_coeff(coeff, iterator, eqn_no, x, negative=False):
    # if(negative):
    #     coeff[0][iterator][eqn_no*4] = -(x**3)
    #     coeff[0][iterator][eqn_no*4+1] = -(x**2)
    #     coeff[0][iterator][eqn_no*4+2] = -(x)
    #     coeff[0][iterator][eqn_no*4+3] = -(1.0)
    # else:
    #     coeff[0][iterator][eqn_no*4] = x**3
    #     coeff[0][iterator][eqn_no*4+1] = x**2
    #     coeff[0][iterator][eqn_no*4+2] = x
    #     coeff[0][iterator][eqn_no*4+3] = 1.0
    factor = -1 if negative else 1
    coeff[0][iterator][eqn_no*4] = (factor)*(x**3)
    coeff[0][iterator][eqn_no*4+1] = (factor)*(x**2)
    coeff[0][iterator][eqn_no*4+2] = (factor)*x
    coeff[0][iterator][eqn_no*4+3] = (factor)*1.0
def set_coeff_first_derivative(coeff, iterator, eqn_no, x, negative=False):
    factor = -1 if negative else 1
    coeff[0][iterator][eqn_no*4] = (factor)*3.0*(x**2)
    coeff[0][iterator][eqn_no*4+1] = (factor)*2.0*x
    coeff[0][iterator][eqn_no*4+2] = (factor)*1.0
    coeff[0][iterator][eqn_no*4+3] = (factor)*0.0
def set_coeff_second_derivative(coeff, iterator, eqn_no, x, negative=False):
    factor = -1 if negative else 1
    coeff[0][iterator][eqn_no*4] = (factor)*6.0*x
    coeff[0][iterator][eqn_no*4+1] = (factor)*2.0
    coeff[0][iterator][eqn_no*4+2] = (factor)*0.0
    coeff[0][iterator][eqn_no*4+3] = (factor)*0.0
    


def interpolate(x, t, base_1=0.0, base_n=0.0):
    '''
    Takes x and t, and interpolates x w.r.t t, and returns an array of coefficients [a,b,c,d]. The equations are solved using matrix multiplication
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




# Main function
def main():
    '''
    Interpolates x and y separately, and then plots them
    '''
    # Set base case for 1st point and last point's first order differential
    base_1=0
    base_n=0

    # Importing waypoint coordinates and initialising equation coefficients matrix
    waypoints = pd.read_csv('loop_track_waypoints.csv')
    waypoints = waypoints.drop('Index', axis=1)
    waypoints = waypoints.to_numpy()
    t = np.zeros((waypoints.shape[0],))
    # Calculating parameter t
    sum=0
    for i in range(waypoints.shape[0]-1):
        t[i] = sum
        sum+=np.sqrt((waypoints[i+1][0] - waypoints[i][0])**2+(waypoints[i+1][1] - waypoints[i][1])**2)
    t[-1] = sum

    coeff_x = interpolate(waypoints[:,0], t)
    coeff_y = interpolate(waypoints[:,1], t)



    # Plotting the interpolated points, x vs t, y vs t
    x_vals = []
    y_vals = []
    t_vals = []
    for i in range(waypoints.shape[0]-1):
        a = [coeff_x[i*4], coeff_y[i*4]]
        b = [coeff_x[i*4+1], coeff_y[i*4+1]]
        c = [coeff_x[i*4+2], coeff_y[i*4+2]]
        d = [coeff_x[i*4+3], coeff_y[i*4+3]]
        
        t_continuous = np.linspace(t[i], t[i+1], 100)
        x_continuous = a[0] * t_continuous**3 + b[0] * t_continuous**2 + c[0] * t_continuous + d[0]
        y_continuous = a[1] * t_continuous**3 + b[1] * t_continuous**2 + c[1] * t_continuous + d[1]
        
        x_vals.extend(x_continuous)
        y_vals.extend(y_continuous)
        t_vals.extend(t_continuous)
    
    plt.plot(x_vals, y_vals, label="Cubic Spline", color="blue",)
    # Plot the original waypoints (from csv filee)
    plt.scatter(waypoints[:,0], waypoints[:,1], color="red", label="Waypoints", zorder=5)
    
    plt.xlabel("x Coordinate")
    plt.ylabel("y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.clf()
    plt.plot(t_vals, x_vals, label="x vs t", color="red")
    plt.scatter(t.tolist(), waypoints[:, 0])
    plt.xlabel("t Parameter")
    plt.ylabel("x Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.clf()
    plt.plot(t_vals, y_vals, label="y vs t", color="green")
    plt.scatter(t.tolist(), waypoints[:, 1])
    plt.xlabel("t Parameter")
    plt.ylabel("y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()








# Entry point
if __name__=='__main__':
    main()