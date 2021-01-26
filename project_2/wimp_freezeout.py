
####### Candidate number - 15331 #######

import numpy as np, matplotlib.pyplot as plt
import scipy.integrate as sc_int
import time as t

N = int(1e3)
m_chi = 1e3             # Gev
S = 1/2                 # spin of WIMP (1/2 for fermions)

# x is a new time variable
x = np.logspace(0,5,N)              # Gev / K
thermal_cs = np.array([1e-9, 1e-10, 1e-11])     # thermally averaged cross sections

# number of internal degrees of freedom of WIMP particle (assuming WIMP are fermions).
# We know only the spin of the WIMP's.
# g_star is effective number of relativistic degrees of freedom (assuming it
# to be constant) and g_star_s is the effective number of relativistic degrees
# of freedom contributing to the entropy density
g = 2*S + 1             # dof for WIMP
g_star = 106.75
g_star_s = 106.75


# definition of the equilibrium value of y
def y_eq(x, mass, dof, dof_star, cs):
    return 4.675e17*dof*mass*cs*x**(3/2)*np.exp(-x) / np.sqrt(dof_star)


# The Boltzmann equation to be solved
def dW_dx(x, W):
    return x**(-2) * (y_eq(x, m_chi, g, g_star, thermal_cs)**2*np.exp(-W) - np.exp(W))



# solving Boltzmann equation
def solve_ODE():
    W0 = np.log(y_eq(x[0],m_chi, g, g_star, thermal_cs))

    # solving the first order differential equation using Radau (implicit) integration method
    solved = sc_int.solve_ivp(dW_dx, t_span=(x[0],x[-1]), y0=W0, method="Radau", t_eval=x)

    W = solved.y
    y = np.exp(W)
    return y


# plots the results of integration the Boltzmann equation with different cross sections
def y_plot(y):
    split = np.where(x>25)[0][0]
    y_eq1 = y_eq(x, m_chi, g, g_star, thermal_cs[0])
    y_eq2 = y_eq(x, m_chi, g, g_star, thermal_cs[1])
    y_eq3 = y_eq(x, m_chi, g, g_star, thermal_cs[2])


    plt.subplot(321)
    plt.title(r"$\left< \sigma v \right>_1 = 10^{-9}$ GeV$^{-2}$")
    plt.plot(x[:split], y[0,:split], "r-", label="y")
    plt.plot(x[:split], y_eq1[:split], "--", color="black", label=r"$y_{eq}$")
    plt.legend(); plt.grid(); plt.yscale("log"); plt.ylabel("y")

    plt.subplot(322)
    plt.title(r"$\left< \sigma v \right>_1 = 10^{-9}$ GeV$^{-2}$")
    plt.plot(x[split:], y[0,split:], "r-", label="y")
    plt.plot(x[split:], y_eq1[split:], "--", color="black", label=r"$y_{eq}$")
    plt.legend(); plt.grid(); plt.xscale("log")


    plt.subplot(323)
    plt.title(r"$\left< \sigma v \right>_2 = 10^{-10}$ GeV$^{-2}$")
    plt.plot(x[:split], y[1,:split], "b-", label="y")
    plt.plot(x[:split], y_eq2[:split], "--", color="black", label=r"$y_{eq}$")
    plt.legend(); plt.grid(); plt.yscale("log"); plt.ylabel("y")

    plt.subplot(324)
    plt.title(r"$\left< \sigma v \right>_2 = 10^{-10}$ GeV$^{-2}$")
    plt.plot(x[split:], y[1,split:], "b-", label="y")
    plt.plot(x[split:], y_eq2[split:], "--", color="black", label=r"$y_{eq}$")
    plt.legend(); plt.grid(); plt.xscale("log")


    plt.subplot(325)
    plt.title(r"$\left< \sigma v \right>_3 = 10^{-11}$ GeV$^{-2}$")
    plt.plot(x[:split], y[0,:split], "g-", label="y")
    plt.plot(x[:split], y_eq1[:split], "--", color="black", label=r"$y_{eq}$")
    plt.legend(); plt.grid(); plt.yscale("log"); plt.xlabel("x"); plt.ylabel("y")

    plt.subplot(326)
    plt.title(r"$\left< \sigma v \right>_3 = 10^{-11}$ GeV$^{-2}$")
    plt.plot(x[split:], y[2,split:], "g-", label="y")
    plt.plot(x[split:], y_eq3[split:], "--", color="black", label=r"$y_{eq}$")
    plt.legend(); plt.grid(); plt.xscale("log"); plt.xlabel("x")

    plt.show()


# calculates dark matter abundance
def dark_matter_abundance(x_dec, dof_star, cs):
    return 1.69e-10*x_dec / (2*np.sqrt(dof_star)*cs)


# calculates the mark matter abundance for specific thermal cross sections
# at the "time" of decoupling
def special_dma_case(time, func, dof_star, cs):
    # finds when the decoupling occurs (when y = 0.1*y_0)
    x_f = time[np.where(func[0] < 0.1*func[0,0])[0]][0],\
          time[np.where(func[1] < 0.1*func[1,0])[0]][0],\
          time[np.where(func[2] < 0.1*func[2,0])[0]][0]
    x_f = np.array(x_f)
    abund = dark_matter_abundance(x_f, dof_star, cs)

    print("Dark matter abundance 1:     %.2f" % abund[0])
    print("Dark matter abundance 2:     %.2f" % abund[1])
    print("Dark matter abundance 3:     %.2f" % abund[2])
    return abund

# calculates the mark matter abundance for continuous thermal cross sections
# at the "time" of decoupling
def general_dma_case(time, dof_star):
    cs_span = np.logspace(-14,-7,N)
    abund_cons = []                 # abundance consistent with observation
    cs_cons = []

    obs = 0.12                      # observed value of the abundace
    lim = 0.05                      # acceptable limit
    high_obs = obs + lim
    low_obs = obs - lim

    global thermal_cs
    for thermal_cs in cs_span:

        W0 = np.log(y_eq(time[0],m_chi, g, g_star, thermal_cs))
        # solving ODE
        solve = sc_int.solve_ivp(dW_dx, t_span=(time[0],time[-1]), y0=[W0], method="Radau", t_eval=time)
        W = solve.y
        y = np.exp(W)

        # finds the time of decoupling
        x_f = time[np.where(y[0] < 0.1*y[0,0])[0][0]]
        abund = dark_matter_abundance(x_f, dof_star, thermal_cs)
        if abund < high_obs+0.01 and abund > low_obs-0.01:
            abund_cons.append(abund)
            cs_cons.append(thermal_cs)
        elif abund < low_obs-0.01:
            break

    plt.plot([cs_cons[0],cs_cons[-1]], [low_obs, low_obs], "--", color="black", label=r"$(\Omega_{dm,0}h^2)_{observed}\pm 0.05$")
    plt.plot([cs_cons[0],cs_cons[-1]], [high_obs, high_obs], "--", color="black")
    plt.plot(cs_cons,abund_cons, color="red", label="Abundance")
    plt.legend(); plt.grid()#; plt.xscale("log")#; plt.yscale("log")
    plt.title("Abundance of dark matter")
    plt.ylabel(r"$\Omega_{dm,0}h^2$"); plt.xlabel(r"$\left< \sigma v \right>$ [GeV$^{-2}]$")
    plt.show()






if __name__ == '__main__':
    print("Solve Boltzmann equation for the special case (given values) of the \
cross sections or for the general case (continuous)? [s/g]:")
    inp = input()

    if inp=="s" or inp=="S":
        y = solve_ODE()
        special_dma_case(x, y, g_star, thermal_cs)
        y_plot(y)

    elif inp=="g" or inp=="G":
        general_dma_case(x, g_star)

    else:
        print("Not an valid input! Try again.")



#
