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
    coeff[0][iterator][eqn_no*4] = (factor)*x**3
    coeff[0][iterator][eqn_no*4+1] = (factor)*x**2
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
    



# Main function
def main():
    '''Algorithm:
    The Matrix coefficients contains calculated coefficients
    Matrices calc_matrix and B_matrix are used in linear algebra equation AX=B. They represent all 4(n-1) equations required to solve the problem
    
    '''
    iterator=0
    # Set base case for 1st point and last point's first order differential
    base_1=0
    base_n=0

    # Importing waypoint coordinates and initialising equation coefficients matrix
    waypoints = pd.read_csv('loop_track_waypoints.csv')
    waypoints = waypoints.drop('Index', axis=1)
    waypoints = waypoints.to_numpy()
    num_eqn = waypoints.shape[0]-1
    coefficients = np.zeros(((num_eqn)*4,)) # ak, bk, ck, dk, will be at index 4*(k-1), 4*(k-1)+1, 4*(k-1)+2, 4*(k-1)+3
    B_matrix = np.zeros(((num_eqn)*4,)) # Initialising 0 matrix at the right side of the equation
    
    # Matrix for calculation
    calc_matrix = [np.zeros(((num_eqn)*4, (num_eqn)*4))] # This will contain num_eqn*4 equations, along with num_eqn*4 coefficients

    # Calculation

    # Boundary Conditions (2nd order)
    set_coeff_second_derivative(calc_matrix, iterator, 0, waypoints[0][0])
    iterator+=1
    set_coeff_second_derivative(calc_matrix, iterator, num_eqn-1, waypoints[num_eqn][0])
    iterator+=1

    for i in range(num_eqn-1):
    # Using coordinates (0th order) and 
        set_coeff(calc_matrix, iterator, i, waypoints[i][0],)
        B_matrix[iterator] = waypoints[i][1]
        iterator+=1
        set_coeff(calc_matrix, iterator, i, waypoints[i+1][0],)
        B_matrix[iterator] = waypoints[i+1][1]
        iterator+=1

    # Equating slopes between two adjacent waypoints (1st order)
        set_coeff_first_derivative(calc_matrix, iterator, i, waypoints[i+1][0])
        set_coeff_first_derivative(calc_matrix, iterator, i+1, waypoints[i+1][0], negative=True)
        iterator+=1
    # Equating slopes of slopes between two adjacent waypoints (2nd order)
        set_coeff_second_derivative(calc_matrix, iterator, i, waypoints[i+1][0])
        set_coeff_second_derivative(calc_matrix, iterator, i+1, waypoints[i+1][0], negative=True)
        iterator+=1


    # These 2 eqns were not included in above loop to avoid the slopes part going out of bound
    set_coeff(calc_matrix, iterator, num_eqn-1, waypoints[num_eqn-1][0],)
    B_matrix[iterator] = waypoints[num_eqn-1][1]
    iterator+=1
    set_coeff(calc_matrix, iterator, num_eqn-1, waypoints[num_eqn-1+1][0],)
    B_matrix[iterator] = waypoints[num_eqn-1+1][1]
    iterator+=1



    # AX = B
    # X = A^-1 x B
    coefficients = np.linalg.solve(calc_matrix[0], B_matrix)

    print(coefficients)


    # Plotting the interpolated points
    fig, ax = plt.subplots()
    x_vals = []
    y_vals = []
    
    for i in range(num_eqn):
        a = coefficients[i*4]
        b = coefficients[i*4+1]
        c = coefficients[i*4+2]
        d = coefficients[i*4+3]
        
        x_segment = np.linspace(waypoints[i][0], waypoints[i+1][0], 100)
        y_segment = a * x_segment**3 + b * x_segment**2 + c * x_segment + d
        
        x_vals.extend(x_segment)
        y_vals.extend(y_segment)
    
    ax.plot(x_vals, y_vals, label="Cubic Spline", color="blue")
    # Plot the original waypoints (from csv filee)
    ax.scatter(waypoints[:,0], waypoints[:,1], color="red", label="Waypoints", zorder=5)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Cubic Spline Interpolation")
    ax.legend()
    ax.grid(True)
    plt.show()

    








# Entry point
if __name__=='__main__':
    main()