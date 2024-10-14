import math

import seaborn as sns
import matplotlib.pyplot as plt



def function1(uo=16):
    """dx/dt = f(x) = -x^2 + sqrt(u)
        Computes the nonlinear and linear forms of the function f(x) = -x^2 + sqrt(u).

        The function calculates the nonlinear form of f(x) and its linear approximation
        around the initial condition x = 2 and u = 16. The linear form is derived using
        the first-order Taylor expansion.

        Parameters:
        uo (int, optional): The parameter u in the function. Default is 16.

        Returns:
        tuple: A tuple containing two lambda functions:
            - f_nonlinear: The nonlinear form of the function.
            - f_linear: The linear approximation of the function.
        """
    print("f(x) = -x^2 + sqrt(u)")

    f_nonlinear = lambda x: -x**2 + math.sqrt(uo)

    # find initial condition of the state by setting the derivative of the function to zero and setting u = 16
    xo = 2
    # function linear form f(x) = f(xo,uo) + f'(xo)(x-xo) + f'(uo)(u-uo)
    f_xo_uo = 0  # f(xo,uo) = 0 since in steady state
    f_prime_xo = -2*xo  # f'(xo) = -2*xo
    f_prime_uo = 1/(2*math.sqrt(uo))  # f'(uo) = 1/(2*sqrt(uo))

    f_linear = lambda x: f_xo_uo + f_prime_xo*(x-xo) + f_prime_uo*(16-uo)
    return f_nonlinear, f_linear

def evaluate_function(f, x):
    """
    Evaluates a given function over a range of input values.

    This function takes a function `f` and a list or range of input values `x`,
    and returns a list of the function's output values for each input in `x`.

    Parameters:
    f (function): The function to be evaluated.
    x (iterable): An iterable of input values.

    Returns:
    list: A list of output values corresponding to each input value in `x`.
    """
    y = []
    for val in x:
        y.append(f(val))
    return y

def plot_function(data):

    """
    Plots the output of a function over a range of input values.

    This function takes a list of lists, where each inner list contains the output
    values of a function evaluated over a range of input values. It then plots each
    function's output values against the input values.

    Parameters:
    data (list): A list of lists, where each inner list contains the output values
        of a function evaluated over a range of input values.
    """
    with sns.color_palette("hls", 8):
        for y in data:
            sns.lineplot(x=range(len(y)), y=y)

    plt.xlabel("Input Value")
    plt.ylabel("Function Value")
    plt.legend(["Nonlinear Function", "Linear Approximation"])
    plt.show()



if __name__ == "__main__":
    f_nonlinear, f_linear = function1()
    y_nonlinear = evaluate_function(f_nonlinear, range(10))
    y_linear = evaluate_function(f_linear, range(10))
    plot_function([y_nonlinear, y_linear])
    pass

