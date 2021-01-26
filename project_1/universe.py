
####### Candidate number - 15331 #######

import matplotlib.pyplot as plt, numpy as np, time as t, sys as s

h = 0.7                             # dimensionless Hubble constant
H0 = 100*h                          # km s^-1 Mpc^-1
c = 3e5                             # km s^-1

print("Size of grid (100 is recommended)?: ")
#N = 200
N = int(input())
if N >499:
    print("Are you sure? This will take some time! (y/n)")
    sure = str(input())
    if sure=="n" or sure=="N":
        s.exit()




file = np.loadtxt("sndata.txt",skiprows=5)
z_data, d_L, err = np.transpose(file)

d_L *= H0/c*1e3
err *= H0/c*1e3


# Friedmann equation for LamCDM model with ohm_r0 = 0
def friedmann_LamCDM(ohm_m0, ohm_k0, ohm_lam0, z):
    fm = ohm_m0*(1+z)**3 + ohm_k0*(1+z)**2 + ohm_lam0
    check = np.where(fm<0)
    if len(check[0]) > 0:
        msg = "The right-hand side of Friedmann (LamCDM) equation is not positive!"
        assert False, msg
    return fm

# Friedmann equation for DGP model
def friedmann_DGP(ohm_m0, ohm_k0, ohm_rc, z):
    fm = (np.sqrt(ohm_m0*(1+z)**3+ohm_rc)+np.sqrt(ohm_rc))**2 + ohm_k0*(1+z)**2
    check = np.where(fm<0)
    if len(check[0]) > 0:
        msg = "The right-hand side of Friedmann (DGP) equation is not positive!"
        assert False, msg
    return fm

# calculating the luminosity distance as a function of the redshift.
# main function in this program
def lum_dist(ohm_m0, ohm_lam0, z, model):

    # checking if the density parametes are single valued
    if type(ohm_m0)==type(1) or type(ohm_m0)==type(1.0):
        # integrating H0/H(z)
        dz_mrk = 1e-3
        z_mrk = np.arange(0,z,dz_mrk)
        if model==LamCDM:
            ohm_k0 = 1 - ohm_m0 - ohm_lam0
            H_over_H0 = np.sqrt(friedmann_LamCDM(ohm_m0, ohm_k0, ohm_lam0, z_mrk))
        elif model==DGP:
            ohm_k0 = 1 - (np.sqrt(ohm_m0+ohm_lam0)+np.sqrt(ohm_lam0))**2
            H_over_H0 = np.sqrt(friedmann_DGP(ohm_m0, ohm_k0, ohm_lam0, z_mrk))
        else:
            msg = "Model not accepted! Insert LamCDM ('$\Lambda$CDM') or DGP ('DGP')"
            assert False,msg
        I = dz_mrk*sum(1/H_over_H0)
        arg = np.sqrt(np.abs(ohm_k0))*I

        if ohm_k0 == 0:
            d_L = (1+z) * I
        elif ohm_k0 < 0:
            d_L = (1+z)/np.sqrt(abs(ohm_k0)) * np.sin(arg)
        else:
            d_L = (1+z)/np.sqrt(abs(ohm_k0)) * np.sinh(arg)

        return d_L

    else:
        # integrating H0/H(z)
        dz_mrk = 1e-3
        z_mrk = np.arange(0,z,dz_mrk)
        H_over_H0 = 0
        if model==LamCDM:
            ohm_k0 = 1 - ohm_m0 - ohm_lam0
            for i in range(len(z_mrk)):
                H_over_H0 += 1/np.sqrt(friedmann_LamCDM(ohm_m0, ohm_k0, ohm_lam0, z_mrk[i]))
        elif model==DGP:
            ohm_k0 = 1 - (np.sqrt(ohm_m0+ohm_lam0)+np.sqrt(ohm_lam0))**2
            for i in range(len(z_mrk)):
                H_over_H0 += 1/np.sqrt(friedmann_DGP(ohm_m0, ohm_k0, ohm_lam0, z_mrk[i]))
        else:
            msg = "Model not accepted! Insert LamCDM ('$\Lambda$CDM') or DGP ('DGP')"
            assert False,msg
        I = dz_mrk*H_over_H0
        arg = np.sqrt(np.abs(ohm_k0)) * I

        # Setting up luminosity distance grid
        mat_len = len(ohm_m0)
        d_L = np.zeros((mat_len,mat_len))

        # finds elements to sustain the S-function (sinx sinhx or x)
        k0_0 = np.where(ohm_k0==0)
        k0_g0 = np.where(ohm_k0>0)
        k0_l0 = np.where(ohm_k0<0)
        j_0 = k0_0[0]
        k_0 = k0_0[1]
        j_g0 = k0_g0[0]
        k_g0 = k0_g0[1]
        j_l0 = k0_l0[0]
        k_l0 = k0_l0[1]
        for i in range(len(j_0)):
            d_L[j_0[i],k_0[i]] = (1+z) * I[j_0[i]][k_0[i]]

        for i in range(len(j_g0)):
            d_L[j_g0[i],k_g0[i]] = (1+z)/np.sqrt(abs(ohm_k0[j_g0[i],k_g0[i]])) * np.sinh(arg[j_g0[i],k_g0[i]])

        for i in range(len(j_l0)):
            d_L[j_l0[i],k_l0[i]] = (1+z)/np.sqrt(abs(ohm_k0[j_l0[i],k_l0[i]])) * np.sin(arg[j_l0[i],k_l0[i]])

        return d_L


# analytical solutions for ohm_k0=ohm_m0=0 and ohm_m0=ohm_lam0=0
def analytical_solutions(red_shift):
    d_L1 = (1+red_shift)*red_shift                          # ohm_k0 = ohm_m0 = 0
    d_L2 = (1+red_shift)*(2-2/np.sqrt(red_shift+1))         # ohm_m0 = ohm_lam0 = 0
    plt.plot(red_shift,d_L1,label="Analytic $d_L$ ($\Omega_{k0} = \Omega_{m0} = 0, \Omega_{\Lambda 0} = 1$)")
    plt.plot(red_shift,d_L2,label="Analytic $d_L$ ($\Omega_{k0} = \Omega_{\Lambda 0} = 0, \Omega_{m0} = 1$)")


# luminocity distance plot for arbitrary Omega_k0 and Omega_lam0
def dL_plot(o_m0, o_lam0, z, model):
    if model==LamCDM:
        o_k0 = 1 - o_m0 - o_lam0
        o_y = "\Lambda 0"
    elif model==DGP:
        o_k0 = 1 - (np.sqrt(o_m0+o_lam0)+np.sqrt(o_lam0))**2
        o_y = "rc"
    dl = np.zeros(len(z))
    for i in range(len(z)):
        dl[i] = lum_dist(o_m0,o_lam0,z[i], model)
    plt.plot(z,dl,label="Numerical $d_L$ (%s) ($\Omega_{k0} = %.2f, \Omega_{m0} = %.2f, \Omega_{%s} = %.2f$)" % (model,o_k0,o_m0,o_y,o_lam0))
    plt.xlabel("Redshift z"); plt.ylabel("Luminosity distance $d_L$ (c/$H_0$)")
    #plt.show()


# calculating chi^2 with grid values of Omega_k0 and
# Omega_lam0 given some redshift value z
def chi_sqrd(z, ohm_m0, ohm_lam0, model):
    chi_squared = 0
    len_z = len(z)
    for i in range(len_z):
        print("Processing a %ix%i grid: %.1f %% " % (N,N,100*i/len_z), end="\r")
        chi_squared += (lum_dist(ohm_m0, ohm_lam0, z[i], model) - d_L[i])**2 / (err[i])**2
    print("Processing a %ix%i grid: %i %% " % (N,N,100))
    return chi_squared


# store the data in a seperate file in case of large grids are runned
def chi_sqrd_data(data_name, model):
    if model==LamCDM:
        Omega_m0 = np.linspace(-0.05,0.6,N)
        Omega_lam0 = np.linspace(0.2,1,N)
        mega = "lambda0"
        method = "LamCDM"
    elif model==DGP:
        Omega_m0 = np.linspace(0,0.5,N)
        Omega_lam0 = np.linspace(0.05,0.3,N)
        mega = "rc"
        method = "DGP"
    OM_m0, OM_lam0 = np.meshgrid(Omega_m0, Omega_lam0, indexing='ij')

    # time the production of the model
    time_a = t.time()
    chi_sq = chi_sqrd(z_data, OM_m0, OM_lam0, model)
    time = t.time()-time_a
    min = time // 60
    sec = time - 60*min
    print("Time: %i:%.2f" % (min,sec))

    # subtracting the smallest value of chi squared
    chi_sq -= np.amin(chi_sq)
    chi_min = np.where(chi_sq==0)
    ohm_m0_best_fit = float(Omega_m0[chi_min[0]])
    ohm_lam0_best_fit = float(Omega_lam0[chi_min[1]])
    print("Most probable parameters (%s): Omega_m0 = %.2f, Omega_%s = %.2f" % (method,ohm_m0_best_fit,mega,ohm_lam0_best_fit))
    print("Data is saved as: %s \n" % data_name)

    data = Omega_m0, Omega_lam0, chi_sq, ohm_m0_best_fit, ohm_lam0_best_fit
    np.save(data_name, data)


# contour plot of the luminosity distance as a function
# of Omega_m0 verus Omega_lam0 for the LamCDM model. Omega_lam0
# is replaced with Omega_rc for the DGP model
def contour_plot(data, model):
    # extracting data from file created in the function chi_sqrd_data
    Omega_m0, Omega_lam0, chi_sq, ohm_m0_best_fit, ohm_lam0_best_fit = np.load(data)
    OM_m0, OM_lam0 = np.meshgrid(Omega_m0, Omega_lam0, indexing='ij')

    # finds the best fit value
    chi_min = np.where(chi_sq==0)
    #levels = [2.3, 6.17, 11.8]
    levels = [6.17]
    plt.contour(OM_m0, OM_lam0, chi_sq, levels=levels, colors='k',label="asf")
    plt.plot(Omega_m0[chi_min[0]], Omega_lam0[chi_min[1]], "r.", label="Most probable")

    if model==LamCDM:
        plt.ylabel("$\Omega_{\Lambda 0}$")
        flat_uni = np.where(abs(1-OM_m0-OM_lam0)==0)
        flat_unix = flat_uni[0]
        flat_uniy = flat_uni[1]
        plt.plot(Omega_m0[flat_unix],Omega_lam0[flat_uniy],label="Flat Universe",color="g")

    if model==DGP:
        plt.ylabel("$\Omega_{rc}$")
        flat_uni = np.where(abs(1 -(np.sqrt(OM_m0+OM_lam0)+np.sqrt(OM_lam0))**2)<1/(3*N))
        flat_unix = flat_uni[0]
        flat_uniy = flat_uni[1]
        plt.plot(Omega_m0[flat_unix],Omega_lam0[flat_uniy],label="Flat Universe",color="g")

    plt.grid(); plt.legend(); plt.xlabel("$\Omega_{m 0}$")
    #plt.show()


LamCDM = "$\Lambda$CDM"
DGP = "DGP"

############ task g), h) ############
print("Run task g and h? (y/n)")
gh = str(input())
if gh=="y" or gh=="Y":
    z = np.linspace(N**(-3),2,N)
    dL_plot(0.12, 0.56, z, LamCDM)              # some random parameters
    dL_plot(0.30, 0.70, z, LamCDM)              # some random parameters
    dL_plot(0, 1, z, LamCDM)                    # to compare with analytic solution
    dL_plot(1, 0, z, LamCDM)                    # to compare with analytic solution
    analytical_solutions(z)
    plt.grid(); plt.legend(); plt.show()


############ task i), j) ############
print("Run task i and j? (y/n)")
ij = str(input())
if ij=="y" or ij=="Y":
    print("Create a new data file? (y/n)")
    new_data = str(input())
    if new_data=="y" or new_data=="Y":
        chi_sqrd_data("Data_LamCDM_%i_pts.npy" % N, LamCDM)
    contour_plot("Data_LamCDM_%i_pts.npy" % N, LamCDM)
    #contour_plot("Data_LamCDM_1500_pts.npy", LamCDM)     # takes about 20 minutes creating this file (N=1500) on my computer
    plt.show()


############ task k) ############
print("Run task k? (y/n)")
k = str(input())
if k=="y" or k=="Y":
    z = np.linspace(N**(-3),z_data[-1],N)
    dL_plot(0.23, 0.62, z, LamCDM)         # best fit parameters (found using contour_plot of LamCDM)
    dL_plot(0.47, 0.89, z, LamCDM)         # random parameters at the edge of the 95 % domain (2 sigma)
    dL_plot(0.02, 0.33, z, LamCDM)         # random parameters at the edge of the 95 % domain (2 sigma)
    dL_plot(1, 0, z, LamCDM)               # dust-filled flat universe (Einstein-de Sitter)
    dL_plot(0, 1, z, LamCDM)               # cosmological constant dominant universe / early universe (de Sitter)
    plt.errorbar(z_data,d_L,yerr=err, label="Data w/errors", fmt=".k")    # data plot
    plt.grid(); plt.legend(); plt.show()


############ task l) ############
print("Run task l? (y/n)")
l = str(input())
if l=="y" or l=="Y":
    print("Create a new data file? (y/n)")
    new_data = str(input())
    if new_data=="y" or new_data=="Y":
        chi_sqrd_data("Data_DGP_%i_pts.npy" % N, DGP)
    contour_plot("Data_DGP_%i_pts.npy" % N, DGP)
    plt.show()

    z = np.linspace(N**(-3),z_data[-1],N)
    dL_plot(0.17, 0.16, z, DGP)             # best fit parameters (found using contour_plot of DGP)
    dL_plot(0.36, 0.23, z, DGP)             # random parameters at the edge of the 95 % domain (2 sigma)
    dL_plot(0.01, 0.08, z, DGP)             # random parameters at the edge of the 95 % domain (2 sigma)
    dL_plot(0, 1/4, z, DGP)                 # to compare with analytic solution
    dL_plot(1, 0, z, DGP)                   # to compare with analytic solution
    analytical_solutions(z)
    plt.errorbar(z_data,d_L,yerr=err, label="Data w/errors", fmt=".k")     # data plot
    plt.grid(); plt.legend(); plt.show()








#
