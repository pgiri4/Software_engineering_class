""" From "COMPUTATIONAL PHYSICS", 3rd Ed, Enlarged Python eTextBook
    by RH Landau, MJ Paez, and CC Bordeianu
    Copyright Wiley-VCH Verlag GmbH & Co. KGaA, Berlin;  Copyright R Landau,
    Oregon State Unv, MJ Paez, Univ Antioquia, C Bordeianu, Univ Bucharest, 2015.
    Support by National Science Foundation"""

# ABM.py:   Adams BM method to integrate ODE
# Solves y' = (t - y)/2,    with y[0] = 1 over [0, 3]


import numpy as np
import matplotlib.pyplot as plt


def get_local_variables(final_calculation = False):
    """
    All required variables are provided from this function
    """
    
    n_steps = 24
    if final_calculation:
        start = 0
        stop = 3
        return n_steps,start,stop
    else:
        independent_var = np.zeros(n_steps+1)
        dependent_var = np.copy(independent_var)
        function_collector = np.zeros(5)
        return independent_var,dependent_var,function_collector


def adams_function(independent_var, dependent_var):
    """
    Returns function of an initial value problem
    """
    return (independent_var - dependent_var) / 2


def runge_kutta(dependent_var, h_step):
    """ 
    Returns y variable from runge kutta method of order 4 
    """
    
    for i in range(3):
        t = h_step * i
        k0 = h_step * adams_function(t, dependent_var[i])
        k1 = h_step * adams_function(t + h_step / 2, dependent_var[i] + k0 / 2)
        k2 = h_step * adams_function(t + h_step / 2, dependent_var[i] + k1 / 2)
        k3 = h_step * adams_function(t + h_step, dependent_var[i] + k2)
        dependent_var[i+1] = dependent_var[i] + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)
    return dependent_var


def adams_bashforth_moulton_method(start, stop, n_steps):
    """    
     Compute 3 additional starting values using runge_kutta
    """

    h_step = (stop - start) / n_steps  # step/size of interval
    h_step2 = h_step / n_steps
    
    independent_var, dependent_var, function_collector = get_local_variables()
    independent_var[0] = start
    dependent_var[0] = 1
    
    for k in range(1, 4):
        independent_var[k] = start + k * h_step
        dependent_var[k] = runge_kutta(dependent_var, h_step)[3]

    for i in range(0,4):
        function_collector[i] = adams_function(independent_var[i], dependent_var[i])
        
    for k in range(3, n_steps):  # Predictor
        p = dependent_var[k] + h_step2 * (-9 * function_collector[0] + 37 * function_collector[1] - 59 * function_collector[2] + 55 * function_collector[3])
        independent_var[k + 1] = start + h_step * (k + 1)  # Next abscissa
        function_collector[4] = adams_function(independent_var[k + 1], p)
        dependent_var[k + 1] = dependent_var[k] + h_step2 * (function_collector[1] - 5 * function_collector[2] + 19 * function_collector[3] + 9 * function_collector[4])  # Corrector
        for i in range(3):
            function_collector[i] = function_collector[i + 1]
        function_collector[3] = adams_function(independent_var[k + 1], dependent_var[k + 1])
    return independent_var, dependent_var


def get_final_result():
    """
    Generate and plot final results
    """
    
    print("  k     t      Y numerical      Y exact")

    n_steps, start, stop = get_local_variables(final_calculation = True)
    independent_var, dependent_var = adams_bashforth_moulton_method(start, stop, n_steps)
    ysol = np.array([3 * np.exp(-tv / 2) - 2 + tv for tv in independent_var])

    for k in range(n_steps + 1):
        print("{: 3d},{: 5.3f},{: 12.11f},{: 12.11f}".format(k, independent_var[k], dependent_var[k], ysol[k]))
        plt.plot(independent_var[: n_steps + 1], dependent_var[: n_steps + 1], "o")
        plt.plot(independent_var[: n_steps + 1], ysol[: n_steps + 1])
    plt.show()
    
    
if __name__ == "__main__" :
    get_final_result()
