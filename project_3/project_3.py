###################################
######### Candidate 15331 #########
###################################


import numpy as np, matplotlib.pyplot as plt

hbar = 1.0545718e-34                        # Plancks constant
c = 299792458                               # speed of light
G = 6.67408                                 # gravitational constant
E_P_sqrd = hbar*c**5/G                      # Planck energy
m_P_sqrd = hbar*c/G                         # Planck mass
l_P_sqrd = hbar*G/c**3                      # Planck length


def potential(phi):
    return E_P_sqrd*phi**2 / (2*1e4*(hbar*c)**3)


phi_i = 11*np.sqrt(E_P_sqrd) / (2*np.sqrt(np.pi))
H_i = np.sqrt(8*np.pi*G*potential(phi_i) / (3*c**2))


def psi(phi):
    return phi/np.sqrt(E_P_sqrd)

def v(psi):
    return E_P_sqrd*psi**2 / (2*1e4*(H_i*hbar)**2)

def dv_dpsi(psi):
    return E_P_sqrd*psi / (1e4*(H_i*hbar)**2)


def inflation_solver():
    N = int(1e4)
    tau_end = 250
    d_tau = tau_end/N
    tau_array = np.linspace(0,tau_end,N)
    psi_array = np.zeros(N)
    h_array = np.zeros(N)
    p_rhoc_array = np.zeros(N)

    psi_array[0] = psi(phi_i)
    d_psi = - dv_dpsi(psi_array[0])/3
    h_array[0] = 1
    p_rhoc_array[0] = (0.5*d_psi**2 - v(psi(phi_i))) / (0.5*d_psi**2 + v(psi(phi_i)))

    # solving the second order equation and the Friedmann equation numerically
    for i in range(1,N):
        dd_psi = - 3*h_array[i-1]*d_psi - dv_dpsi(psi_array[i-1])
        d_psi = d_psi + dd_psi*d_tau
        psi_array[i] = psi_array[i-1] + d_psi*d_tau

        vv = v(psi_array[i])
        h_array[i] = np.sqrt(abs(8*np.pi * (0.5*d_psi**2 + vv)/3))
        p_rhoc_array[i] = (0.5*d_psi**2 - vv) / (0.5*d_psi**2 + vv)

    return tau_array, psi_array, h_array, p_rhoc_array


# analytic solution of phi from the compendium
def analytical_phi(t):
    return phi_i - E_P_sqrd*t / (1e2*hbar*np.sqrt(12*np.pi))


tau, psi_num, h, p_rhoc = inflation_solver()
psi_an = psi(analytical_phi(tau/H_i))

plt.rcParams.update({'font.size': 12})
plt.figure(1)
plt.plot(tau, psi_num, label="Numerical")
plt.plot(tau, psi_an, label="Analytical")
plt.legend(); plt.grid(); plt.title("Scalar field")
plt.xlabel(r"$\tau$"); plt.ylabel(r"$\psi$")
#plt.show()

# integrating h with respect to tau from zero to tau
h_int = np.zeros(len(tau))
d_tau = tau[-1]/len(tau)
for i in range(1,len(tau)):
    h_int[i] = h_int[i-1] + h[i]*d_tau

plt.figure(2)
plt.plot(tau,h_int)
plt.xlabel(r"$\tau$"); plt.ylabel(r"$\ln(a/a_i)$")
plt.grid(); plt.title(r"$e$-foldings")
print("e-foldings: ", h_int[-1])
#plt.show()

plt.figure(3)
plt.plot(tau, p_rhoc)
plt.xlabel(r"$\tau$"); plt.ylabel(r"$p_{\phi} / \rho_{\phi}c^2$")
plt.grid(); plt.title("Scalar field pressure/energy field ratio")
plt.show()
