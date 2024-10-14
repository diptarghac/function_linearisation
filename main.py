import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint


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

    f_nonlinear = lambda x, u: -x ** 2 + math.sqrt(u)

    # find initial condition of the state by setting the derivative of the function to zero and setting u = 16
    xo = 2
    # function linear form f(x) = f(xo,uo) + f'(xo)(x-xo) + f'(uo)(u-uo)
    f_xo_uo = 0  # f(xo,uo) = 0 since in steady state
    f_prime_xo = -2 * xo  # f'(xo) = -2*xo
    f_prime_uo = 1 / (2 * math.sqrt(uo))  # f'(uo) = 1/(2*sqrt(uo))

    f_linear = lambda x, u: f_xo_uo + f_prime_xo * (x - xo) + f_prime_uo * (u - 16)
    return f_nonlinear, f_linear


def evaluate_function(f, x, input=0):
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
        y.append(f(val, input))
    return y


def plot_function(data, independent_variable):
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
            sns.lineplot(x=independent_variable, y=y)

    plt.xlabel("Input Value")
    plt.ylabel("Function Value")
    plt.legend(["Nonlinear Function", "Linear Approximation"])
    plt.show()


def plot_state_variable(time, x1, x2, u):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(time, u, 'g-', linewidth=3, label='u(t) Input step change')
    plt.grid()
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(time, x1, 'b-', linewidth=3, label='x(t) Nonlinear')
    plt.plot(time, x2, 'r--', linewidth=3, label='x(t) Linear')
    plt.xlabel('time')
    plt.grid()
    plt.legend(loc='best')
    plt.show()


def model(z, t, u, f_linear, f_nonlinear):
    x1 = z[0]
    x2 = z[1]
    dx1dt = f_nonlinear(x1, u)
    dx2dt = f_linear(x2, u)
    dzdt = [dx1dt, dx2dt]
    return dzdt


def step_disturbance(tdist1, tdist2, tdist3, time, input_ss, step=8):
    u = input_ss * np.ones_like(time)

    indx1 = int(np.where(time == time[(time > tdist1) & (time < tdist1 + 0.02)])[0])
    indx2 = int(np.where(time == time[(time > tdist2) & (time < tdist2 + 0.05)])[0])
    indx3 = int(np.where(time == time[(time > tdist3) & (time < tdist3 + 0.08)])[0])

    # change up m at time = 1.0
    u[indx1:] = u[indx1:] + step
    # change down 2*m at time = 4.0
    u[indx2:] = u[indx2:] - 2.0 * step
    # change up m at time = 7.0
    u[indx3:] = u[indx3:] + step

    return u


def solve_diff_eq(time, u, z_ic, f_linear, f_nonlinear):
    x1 = np.empty_like(time)
    x2 = np.empty_like(time)
    for i in range(1, len(time)):
        dt = [time[i - 1], time[i]]
        z = odeint(model, z_ic, dt, args=(u[i], f_linear, f_nonlinear))
        z_ic = z[1]
        x1[i] = z[1][0]
        x2[i] = z[1][1]
    return x1, x2


if __name__ == "__main__":
    state_variable_sweep = np.linspace(1.5, 3, 10)
    input_ss = 16
    state_variable_ss = 2

    f_nonlinear, f_linear = function1()
    y_nonlinear = evaluate_function(f_nonlinear, state_variable_sweep, input=input_ss)
    y_linear = evaluate_function(f_linear, state_variable_sweep, input=input_ss)
    # plot_function([y_nonlinear, y_linear], state_variable_sweep)

    z_ic = [state_variable_ss, state_variable_ss]
    time = np.linspace(0, 10, 100)
    tdist1, tdist2, tdist3 = 1, 4, 7
    u = step_disturbance(tdist1, tdist2, tdist3, time, input_ss)
    # store ODE
    x1, x2 = solve_diff_eq(time, u, z_ic, f_linear, f_nonlinear)
    plot_state_variable(time, x1, x2, u)

    pass
