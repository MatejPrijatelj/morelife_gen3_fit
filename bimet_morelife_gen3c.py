#0D Pt/Co catalyst degradation model
### library inports

#system libraries
import sys
import os
import time
import pathlib as pth
from datetime import datetime
import signal

#numpy
import numpy as np

#pandas
import pandas as pd

#ODE solver
from scipy.integrate import solve_ivp
from scipy.integrate import dblquad


#optimisation and fiting
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy.optimize import differential_evolution

#distribution
from scipy.stats import norm, lognorm, multivariate_normal

#numba
from numba import jit, config, set_num_threads, get_num_threads,threading_layer

import matplotlib.pyplot as plt

### numba settings


isparalel=False
#disable jit
config.DISABLE_JIT = False
#se trhreading layer
config.THREADING_LAYER = 'omp'
#config.THREADING_LAYER = 'tbb'


#SET FOR OPTIMISATION!!!
config.NUMBA_NUM_THREADS = 1
config.NUMBA_DEFAULT_NUM_THREADS = 1
set_num_threads(1)
#disable numpy threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


#config.NUMBA_NUM_THREADS=1
#set number of threads
#SET FOR OPTIMISATION
#set_num_threads(1)
#threading_layer('omp')


### file inport/outport
#here we inport fiting data, set output file path etc.

#get path
mypath=os.getcwd()
print( "working from dir: " + str(pth.Path(mypath)))

#import voltage profile for one cycle if used
#data=np.loadtxt("UcatData_RealCycCEA.txt",skiprows=2)
#r_bounds=data[:,0]
#np.insert(r_bounds,0, 0)
#r_dist=data[:,1]

### Define global constants
R           =   8.314           #J/K*mol
F           =   96485           #As/mol
eps0        =   8.85*10**-12    #F/m
rhoPt       =   21450*10**3     #g/m^3 pt mass density
molarPt     =   195.084         #Pt molar mass in g/mol
molarCu     =   63.5            #Cu molar mass i g/mol
Pt_mol_mas  =   195.084         # molarPt  Pt molar mass in g/mol
Co_mol_mas  =   63.5            # molarCo  Co molar mass in g/mol
Pt_density  =   21450*10**3     # rhoPt g/m^3

OmegaPt     =   9.09*10**-6     # OmegaPt Pt molar volume [m3/mol]
OmegaCu     =   9.09*10**-6     # OmegaCu   Co/Pt molar volume [m3/mol] its probably more lie 11.6 1e-6 a sigh

GamaPt      =   2.2*10**-5             #pt surface site density [mol/m2] #added since its a constant

#r0          =   10.*10**-9#Pt particle diameter in m

### system parameters
#these are edited in optimisation
ModPars=np.zeros(29)

#oxide kinetic parameters
ModPars[0]=3.0*10**4    # nuForPtO  Forward Pt oxide rate [1/s] 
ModPars[1]=6.0*10**-2   # nuBacPtO  backward Pt oxide rate [1/s] 
ModPars[2]=2.2*10**-5   # GamaPt    Pt surface site density [mol/m2] 
ModPars[3]=1.2*10**4    # HEnthPtO  Partial molar oxide formation enthalpy (0 coverage) [J/mol]
ModPars[4]=0.5          # betPtO    Butler-Volmer transfer coefficient; PtO formation []
ModPars[5]=2.0          # nElPtO    electrons transfered during Pt oxidation []
ModPars[6]=1.03         # UPtObulk  PtO eq voltage [V]
ModPars[7]=2.0*10**4    # lamPtO    PtO dependant kinetic barier constant [J/mol]
ModPars[8]=5.0*10**4    # omegaPto  PtO-PtO interaction energy [J/mol]

#pt kinetic parameters
ModPars[9]=1.0*10**4    # nuForPt   Dissolution atmpto frequency [1/s]
ModPars[10]=8*10**5     # nuBacPt   back dissolution rate [1/s]
ModPars[11]=4.5*10**4   # HEnthPt   Partial molar formation enthalpy (0 coverage) [J/mol]
ModPars[12]=0.5         # betPt     Butler-Volmer transfer coefficient Pt []
ModPars[13]=2.0         # nElPt     Pt dissolution electron transfer []
ModPars[14]=1.118       # UPt       Equelibrium voltage Pt dissolution [V]

#surface tension, ref conc,  molar volume
#ModPars[15]=9.09*10**-6 # OmegaPt   Pt molar volume [m3/mol]
ModPars[16]=2.4         # gamma     Pt [111] surface tension [J/m2]
ModPars[17]=1.0*10**-3  # CPtRef    reference Pt conc  [mol/l]

#cell parameters , surface tension orrection factor
ModPars[18]=0.4         # xiH2O     volume fraction of water in ionomer
ModPars[19]=0.2         # muNaf     volume fraction of ionomer in cathode electrode
ModPars[20]=20.*10**-6  # L         CCL thickness [m]
ModPars[21]=1.0*10**-2  # A         cell surface area [m^2] TODO not really needed.
ModPars[22]=0.02*10**9  # eps_u     correction factor [V/m]
ModPars[23]=0.15*10**1   # PtLoad    Pt ploading [g/m^2] this is recalculated to particle number, but is a better parameter

#aloying metal end composiiton
#ModPars[24]=9.09*10**-6 # OmegaCu   Co/Pt molar volume [m3/mol] its probably more lie 11.6 1e-6 a sight chage
ModPars[25]=3         # n         Pt/Cu ratio []

Pars=["Forward Pt oxide rate [1/s]"]
Pars.append("backward Pt oxide rate [1/s]")
Pars.append("Pt surface site density [mol/m2]")
Pars.append("Partial molar oxide formation enthalpy (0 coverage) [J/mol]")
Pars.append("Butler-Volmer transfer coefficient; PtO formation []")
Pars.append("electrons transfered during Pt oxidation []")
Pars.append("PtO eq voltage [V]")
Pars.append("PtO dependant kinetic barier constant [J/mol]")
Pars.append("PtO-PtO interaction energy [J/mol]")
    
Pars.append("Dissolution atmpto frequency [1/s]")
Pars.append("back dissolution rate [1/s]")
Pars.append("Partial molar formation enthalpy (0 coverage) [J/mol]")
Pars.append("Butler-Volmer transfer coefficient Pt []")
Pars.append("Pt dissolution electron transfer []")
Pars.append("Equelibrium voltage Pt dissolution [V]")
    
Pars.append("Pt molar volume [m3/mol]")
Pars.append("Pt [111] surface tension [J/m2]")
Pars.append("reference Pt conc  [mol/l]")
Pars.append("volume fraction of water in ionomer []")
Pars.append("volume fraction of ionomer in cathode electrode []")
Pars.append("CCL thickness [m]")
Pars.append("cell surface area [m^2]")
Pars.append("corect factor [V/m]")
Pars.append("Pt ploading [g/m^2] this is recalculated to particle number, but is a better parameter")

Pars.append("Cu/Pt molar volume [m3/mol]")
Pars.append("Cu/Pt ratio []")


###input parameter functions
#temperature, pH, Pt_Voltage

#temperature
T0=273+80
#cH+
Hp0=10**(0)
#pt voltage
Umin=0.6
Umax=1.0

###input functions.

# function that returns interpolated voltage profile.
#@jit(nopython=True,parallel=False)
#def funcUPt_int(x_vals):
#    return np.interp(x_vals % 1200, data[:,-1], data[:,0])

# constant voltage funxtion
@jit(nopython=True,parallel=isparalel)
def funcUPt_const(x):
    return Umax

@jit(nopython=True)
def funcUPt_Morelife(x):
    return (2.*Am/p)*abs( (x-p/4.)%p - p/2.) + Umin


# AST voltage profile
@jit(nopython=True)
def funcUPt_saw(x):
    return (2.*Am/p)*abs( (x-p/4.)%p - p/2.) + Umin

# Temperature, constant, but can be modified
@jit(nopython=True,parallel=isparalel)
def funcT(x):
    return T0

# pH constant, can be modified
@jit(nopython=True,parallel=isparalel)
def funccHp(x):
    return Hp0

# Heaviside function. it is used in preparation of RHS (space) derivatives
@jit(nopython=True,parallel=isparalel)
def f_nb_heaviside(x):
    if x>=0:
        return 1.
    else:
        return 0.

#helper functions not involved in main simulations
#for this reason these functions are not JITed

#transforms weight ratios to atomic ratios
#variables:
# weight_ratios = list of ratios
# molar_masses = list of molar mases

def f_weight_to_atomic_ratio(weight_ratios, molar_masses):
    
    # Calculate the number of moles for each element
    moles = [weight / molar_mass for weight, molar_mass in zip(weight_ratios, molar_masses)]

    # Find the total number of moles
    total_moles = sum(moles)

    # Calculate the atomic ratio for each element
    atomic_ratios = [mole / total_moles for mole in moles]

    return atomic_ratios

#transforms atomic ratios to weight ratios
#variables:
# atomic_ratios = list of ratios
# molar_masses = list of molar mases

def f_atomic_to_weight_ratio(atomic_ratios, molar_masses):

    # Calculate the weight of each element based on atomic ratios and molar masses
    weights = [atomic_ratio * molar_mass for atomic_ratio, molar_mass in zip(atomic_ratios, molar_masses)]

    # Find the total weight
    total_weight = sum(weights)

    # Calculate the weight ratio for each element
    weight_ratios = [(weight / total_weight) * 100 for weight in weights]

    return weight_ratios

# transfrom from 2d distribution to X vector
#variables:
#dist_bins_prob = 2d PSD (m X n)
#start = [0,0,0] Solute concentrations vectors
def f_mat2vec(dist_bins_prob,start=[0,0,0]):
    l=len(start)
    m,n=np.shape(dist_bins_prob)
    X=np.zeros(l+m*n)
    X[l:]=dist_bins_prob.flatten()
    X[:l]=start

    return X

# function to weight contribution of thin Pt shell to activity.
# return 1 for values larger than 6M and linear trend for smaller radii
#variables:
#r = particle radius in m

def f_fun_fact(l,A=3.):
    #A=3.
    ml=0.25*10**-9
    lim=6
    
    if 0.< l < ml:
        return A
    elif ml <= l <= lim*ml:
        # Linear decrease from A (at m_l) to 1 (at M_l)
        return A - (A - 1) * (l - ml) / (lim*ml - ml)
    else:
        return 1.

# transform X vetor to 2D PSD
#variables:
#X input vector size nst+len(v)*len(c)
#vec_l vector of shell bin mids
#vec_c vector of core bin mids
#nSt  number of variables in Pt,Pt0, Co
def f_X_to_dist(X,vec_l,vec_c,nSt=3):
    nc=len(vec_c)
    nl=len(vec_l)
    XX=np.zeros((nl,nc))
    
    ### fill the XX matrix (more convinient) with  X input vector
    for i in range(nl): 
        XX[i,:]=X[nSt+nc*i:nSt+nc*(i+1)]
    return XX

# calculate crude mean of distribution in r
#inputs:
#om PSD shape (len(v),len(c))
#vec_l vector of shell bin mids
#vec_c vector of core bin mids
def f_mean(om,vec_l,vec_c):
    #transform vector to distribution
    #total weights
    tot_om=np.sum(om)
    
    r_mean=0
    for i,l in enumerate(vec_l):
        for j,c in enumerate(vec_c):
            r_mean = r_mean + (l+c)*om[i,j]/tot_om       
    return r_mean

# calculate crude standard dev of distribution in r 
#inputs:
#om PSD shape (len(v),len(c))
#vec_l vector of shell bin mids
#vec_c vector of core bin mids
 
def f_stdev(om,vec_l,vec_c):    
    #transform vector to distribution
    #total weights
    tot_om=np.sum(om)
    r_mean=f_mean(om,vec_l,vec_c)
    
    r_std=0
    for i,l in enumerate(vec_l):
        for j,c in enumerate(vec_c):
            r_std = r_std + (l+c-r_mean)**2*om[i,j]/tot_om
    
    r_std=np.sqrt(r_std)       
    return r_std

# surface area per bin and weighted active surface per bin
# usefull in order to evaluate bin importance to electrochemical activity
#inputs:
#om PSD shape (len(v),len(c))
#vec_l vector of shell bin mids
#vec_c vector of core bin mids

def f_om_to_surf(om,vec_c,vec_l,fun_fact = f_fun_fact):
    #copy input to output, they are the same shape
    surf_om = np.copy(om)
    act_om = np.copy(om)
    
    Dc=vec_c[1]-vec_c[0]
    Dl=vec_l[1]-vec_l[0]
    
    for i,l in enumerate(vec_l):
        for j,c in enumerate(vec_c):
            r=c+l
            num=Dc*Dl*om[i,j]
            #v_core  = num * (4./3.) * 3.14 * c**3
            #v_shell = num * (4./3.) * 3.14 * r**3 - v_core
            S = num * 4 * 3.14 *  r**2
            fact = fun_fact(l)
            surf_om[i,j] = S
            act_om[i,j] = S*fact
    #loop trough core and shell, 
    return surf_om,act_om

# transform core and shell volume to Pt and Cu in mols
#inputs:
#X input vector size nst+len(v)*len(c)
#vec_l vector of shell volume
#vec_c vector of core volume
#ModPars the parameters vector
def f_vol_to_PtCu(c_tot,l_tot,ModPars):
    
    n           =   ModPars[25]
    #OmegaPt     =   ModPars[15]
    #OmegaCu     =   ModPars[24]
    
    Pt = (l_tot / OmegaPt) + ( c_tot / OmegaCu ) * (n/(n+1.))
    Cu = (c_tot / OmegaCu) * (1./(n+1.))
    
    return Pt,Cu

#returns surface area of a particle with given radius
def f_SA_from_r(r):
    return 4*np.pi*r**2

#returns volume of a particle with given radius
def f_VOL_from_r(r):
    return (4./3.)*np.pi*r**3

#returns volume of core and shell from 
def f_vols_from_cl(c,l):
    r=l+c
    vc  =  f_VOL_from_r(c)
    vl =   f_VOL_from_r(r) - vc
    return vc,vl

#returns concentration d noramalised to cell cathode layer from mols of metal
def f_n_to_c(c,muNaf,L,A):
    return c*10**-3/(muNaf*L*A)
    
#returns a ratio of Pt to co from core and shell
def f_c_l_to_ratio(c,l,MPars):
    f_VOL_from_r

# returns various statistic values from the PSD
# returns average radius, core and shell, ponderated surface area.
# actual surface area total core and shell volume, 
# total Pt and Cu inn remaining particles transformed to concentration if disolved
#inputs:
#ModPars the parameters vector
#vec_l vector of shell bin mids
#vec_c vector of core bin mids

#returns geoe area per cm2 from geo area per g Pt
def f_mas_S_to_surf_S(S,ModPars):
    L   =   ModPars[20] #CCL thickness [m]
    A   =   ModPars[21] #cell surface area [m^2]
    load = ModPars[23]  #0.4*10**1   # PtLoad    Pt ploading [g/m^2]
    return S

def f_stats_old(X,ModPars,dist_cs,dist_ls):
    
    Dc=dist_cs[1]-dist_cs[0]
    Dl=dist_ls[1]-dist_ls[0]

    nc=len(dist_cs)
    nl=len(dist_ls) #why dont i need it

    nSt=3
    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]
    load        =   ModPars[23] #0.4*10**1   # PtLoad    Pt ploading [g/m^2] this is recalculated to particle number, but is a better parameter

    SA = 0.
    ESA = 0.
    r_avg = 0.
    c_avg = 0.
    l_avg = 0.
    #total particle number
    num_tot = 0.
  
    #calculating total platinum content
    #Pt = 0.
    #calculating total coper content
    #Cu = 0.
    
    #calculating total surface
    #S_tot = 0
    #calculating total core volume
    c_tot = 0
    #calculating total shell volume
    l_tot = 0
    
    for i,l in enumerate(dist_ls):
        for j,c in enumerate(dist_cs):
            r=c+l
            num=Dc*Dl*X[nSt+i*nc+j] #correct indexing
            
            #calc core  and shell volume and surface of each group
            v_core  =   f_VOL_from_r(c)
            v_shell =   f_VOL_from_r(r) - v_core
            S       =   f_SA_from_r(r)

            #electrochemical factor function accounting for speciffic activity gains associated with thin shells
            fact = f_fun_fact(l)
            
            #S_tot=S_tot + num * S

            c_tot=c_tot + num * v_core
            l_tot=l_tot + num * v_shell

            SA = SA + num *S
            ESA = ESA + num *S*fact
            num_tot = num_tot + num
            r_avg = r_avg + num * r
            c_avg = c_avg + num * c
            l_avg = l_avg + num * l

    r_avg=r_avg/num_tot
    c_avg=c_avg/num_tot
    l_avg=l_avg/num_tot

    #recalculate Surface area to S per g Pt
    #calculate Pt and cu mols from volume of core, shell and parameters
    Ptm,Cum=f_vol_to_PtCu(c_tot,l_tot,ModPars)
    #Pt = (l_tot / OmegaPt) + ( c_tot / OmegaCu ) * (n/(n+1.))
    #Cu = (c_tot / OmegaCu) * (1./(n+1.))

    #Transform Surface to m2/g  should be values around 50-70 for 2nm particles
    #total Pt masi in cell area
    M=load*A
    
    SA=SA/M
    ESA=ESA/M
    
    #transform volume to concentrationPt
    #makes more sense if we are dealing with disolution 
    
    #Pt=Pt*10**-3/(muNaf*L*A)
    Ptc=f_n_to_c(Ptm,muNaf,L,A)
    #Cu=Cu*10**-3/(muNaf*L*A)
    Cuc=f_n_to_c(Cum,muNaf,L,A)
    #change to conentrations 
    #Pt=Pt*10**-3/(muNaf*L)
    #Cu=Cu*10**-3/(muNaf*L)    
    
    return (num_tot,r_avg,c_avg,l_avg,SA,ESA,c_tot,l_tot,Ptc,Cuc)

#returns some geometric quantities for a single particle as well as activity
def f_part_stats(c,l,ffun=f_fun_fact):
    #special case if we are dealing with pure platinum as well
    if c>0.:
        r=c+l
        fact = ffun(l)
    else:
        r=l
        c=0.
        fact=1.
                
    #num=Dc*Dl*om[i,j] #correct indexing
    #calc core  and shell volume and surface of each group
    v_core  =   f_VOL_from_r(c)
    v_shell =   f_VOL_from_r(r) - v_core
    S       =   f_SA_from_r(r)
                
    S_t=S
    ES_t=S*fact
        #electrochemical factor function accounting for speciffic activity gains associated with thin shells
                
        #S_tot=S_tot + num * S
    return r,c,l,S_t,ES_t,v_core,v_shell

#retruns f_part_stats on distribution. it works on X for historicla reasons.
def f_part_stats_dist(X,ModPars,dist_cs,dist_ls):
    
    omS=np.zeros((len(dist_ls),len(dist_cs)))
    omESA=np.zeros((len(dist_ls),len(dist_cs)))
    
    #print(np.shape(om))
    
    for i,l in enumerate(dist_ls):
        for j,c in enumerate(dist_cs):
            #num=1.
            num=f_return_num(X,ModPars,dist_cs,dist_ls,i,j)
            r,c,l,S_t,ES_t,v_core,v_shell = f_part_stats(c,l)
            #print(str(i)+" "+str(j))
            omS[i,j]=num*S_t
            omESA[i,j]=num*ES_t
            
    return omS,omESA

#returns number of particles in
def f_return_num(X,ModPars,dist_cs,dist_ls,i,j):    
    Dc=dist_cs[2]-dist_cs[1]
    Dl=dist_ls[2]-dist_ls[1]
    nc=len(dist_cs)
    
    nSt=3
    num=Dc*Dl*X[nSt+i*nc+j] 
    return num

# returns various statistic values from the PSD
# returns average radius, core and shell, ponderated surface area.
# actual surface area total core and shell volume, 
# total Pt and Cu inn remaining particles transformed to concentration if disolved
#inputs:
#ModPars the parameters vector
#vec_l vector of shell bin mids
#vec_c vector of core bin mids
#updated geometric and electrochemical surface are nw normalised to g pt loading.
def f_stats(X,ModPars,dist_cs,dist_ls,ffun=f_fun_fact):

    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]
    load        =   ModPars[23] #0.4*10**1   # PtLoad    Pt ploading [g/m^2] this is recalculated to particle number, but is a better parameter

    SA = 0.
    ESA = 0.
    r_avg = 0.
    c_avg = 0.
    l_avg = 0.
    #total particle number
    num_tot = 0.
    
    #calculating total platinum content
    #Pt = 0.
    #calculating total coper content
    #Cu = 0.
    
    #calculating total surface
    #S_tot = 0
    #calculating total core volume
    c_tot = 0
    #calculating total shell volume
    l_tot = 0
    
    #om=f_X_to_dist(X,dist_ls,dist_cs)

    for i,l in enumerate(dist_ls):
        for j,c in enumerate(dist_cs):
            num=f_return_num(X,ModPars,dist_cs,dist_ls,i,j)
            r,c,l,S_t,ES_t,v_core,v_shell = f_part_stats(c,l,ffun)

            c_tot=c_tot + num * v_core
            l_tot=l_tot + num * v_shell

            SA = SA + S_t*num
            ESA = ESA +  ES_t*num
            num_tot = num_tot + num
            r_avg = r_avg + num * r
            c_avg = c_avg + num * c
            l_avg = l_avg + num * l
           
    r_avg=r_avg/num_tot
    c_avg=c_avg/num_tot
    l_avg=l_avg/num_tot


    #recalculate Surface area to S per g Pt
    #calculate Pt and cu mols from volume of core, shell and parameters
    Ptm,Cum=f_vol_to_PtCu(c_tot,l_tot,ModPars)
    #Pt = (l_tot / OmegaPt) + ( c_tot / OmegaCu ) * (n/(n+1.))
    #Cu = (c_tot / OmegaCu) * (1./(n+1.))

    #Transform Surface to m2/g  should be values around 50-70 for 2nm particles
    #total Pt masi in cell area
    M=load*A
    
    SA=SA/M
    ESA=ESA/M
    
    #transform volume to concentrationPt
    #makes more sense if we are dealing with disolution 
    #Pt=Pt*10**-3/(muNaf*L*A)
    Ptc=f_n_to_c(Ptm,muNaf,L,A)
    #Cu=Cu*10**-3/(muNaf*L*A)
    Cuc=f_n_to_c(Cum,muNaf,L,A)
    #change to conentrations 
    #Pt=Pt*10**-3/(muNaf*L)
    #Cu=Cu*10**-3/(muNaf*L)
    
    return (num_tot,r_avg,c_avg,l_avg,SA,ESA,c_tot,l_tot,Ptc,Cuc)

#alternative f_stats that loops trough dist. nested.
def f_stats_1(X,ModPars,dist_cs,dist_ls):
    
    nSt=3
    om=f_X_to_dist(X,vec_l,vec_c,nSt)
    
    Dc=dist_cs[1]-dist_cs[0]
    Dl=dist_ls[1]-dist_ls[0]

    n           =   ModPars[25] #Pt/Cu
    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]
    #OmegaPt     =   ModPars[15]
    #OmegaCu     =   ModPars[24]
 
     
    SA = 0.
    ESA = 0.
    r_avg = 0.
    c_avg = 0.
    l_avg = 0.
    #total particle number
    num_tot = 0.
  
    #calculating total platinum content
    Pt = 0.
    #calculating total coper content
    Cu = 0.
    
    #calculating total surface
    S_tot = 0
    #calculating total core volume
    c_tot = 0
    #calculating total shell volume
    l_tot = 0
    
    for i,l in enumerate(dist_ls):
        for j,c in enumerate(dist_cs):
            r=c+l
            num=Dc*Dl*om[i,j] #correct indexing
            
            v_core  = num * (4./3.) * 3.14 * c**3
            v_shell = num * (4./3.) * 3.14 * r**3 - v_core
            S = num * 4 * 3.14 *  r**2
            fact = f_fun_fact(l)
            
            S_tot=S_tot+S

            c_tot=c_tot+v_core
            l_tot=l_tot+v_shell

            SA = SA + S
            ESA = ESA + S*fact
            num_tot = num_tot + num
            r_avg = r_avg + num * r
            c_avg = c_avg + num * c
            l_avg = l_avg + num * l

    r_avg=r_avg/num_tot
    c_avg=c_avg/num_tot
    l_avg=l_avg/num_tot

    Pt = (l_tot / OmegaPt) + ( c_tot / OmegaCu ) * (n/(n+1.))
    Cu = (c_tot / OmegaCu) * (1./(n+1.))

    #transform volume to concentrationPt
    #makes more sense if we are dealing with disolution 
    Pt=Pt*10**-3/(muNaf*L*A)
    Cu=Cu*10**-3/(muNaf*L*A)
    #change to conentrations 
    #Pt=Pt*10**-3/(muNaf*L)
    #Cu=Cu*10**-3/(muNaf*L)    
    
    return (num_tot,r_avg,c_avg,l_avg,SA,ESA,c_tot,l_tot,Pt,Cu)


#returns distribution in shells (vector length len(dist_ls))
#variables:
#X input vector size nst+len(v)*len(c)
#vec_l vector of shell bin mids
#vec_c vector of core bin mids
def f_l_S_dist(X,dist_cs,dist_ls):
    
    Dc=dist_cs[1]-dist_cs[0] 
    Dl=dist_ls[1]-dist_ls[0]
    
    nc=len(vec_c)
    nl=len(vec_l)
    
    rez_l_S=np.zeros(len(dist_ls))

    for i,l in enumerate(dist_ls):
        Si=0
        for j,c in enumerate(dist_cs):
            r = c + l
            num = Dc*Dl*X[3+i*nc+j]
            Si = Si + 4 * 3.14 * num * r**2
        rez_l_S[i]=Si
        
    return rez_l_S


#retrurns Pt conc
#variables:
#ModPars=model parameters
def f_Pt_conc(ModPars):
    Ptc=ModPars[23]*1e-3/(Pt_mol_mas*ModPars[20]*ModPars[19])
    return Ptc

### returns starting particle distribution
#variables:
#lmax = max shellvalue
#cmax = max core value
#nl = number of shell bins
#nc = number of core bins
#MPars = model parameter
def f_start_dist(lmax=5,cmax=5,nl=20,nc=20,MPars=ModPars):

    n           =   ModPars[25] #Pt/Cu
    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]
    #OmegaPt     =   ModPars[15]
    #OmegaCu     =   ModPars[24]

    PtMass=MPars[23]*MPars[21] # Mass Of platinum in g caluclated by loading and surface area
    VolCat=MPars[20]*MPars[21]*MPars[19] #cathode volume
    NPtTot=PtMass/molarPt #mols of platinum

    #bin borders
    vec_c_borders=np.linspace(0, cmax, nl)
    vec_l_borders=np.linspace(0, lmax, nc)

    #number of bins
    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    vec_c=np.zeros(Nc)
    vec_l=np.zeros(Nl)

    #fill bin vectors
    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5

    #grid to ease construction    
    C,L = np.meshgrid(vec_c,vec_l)

    #variables governing distribution
    val=0
    var1=1
    var2=0.1
    mean = np.array([3,0.25])

    cov = np.array([[var1, val], [val, var2]])
    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr = multivariate_normal(cov = cov, mean = mean)
    distr_l = norm(loc=0.25, scale=0.25)
    distr_c = lognorm(s=0.7,loc=0.4, scale=2)

    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pdf[i,j] = distr_l.pdf(L[i,j])*distr_c.pdf(C[i,j])

    #empty the first bin for stability reasons
    pdf[0,0]=0
    #scale to nanometers
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    dist_bins_prob=pdf

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    X = f_mat2vec(dist_bins_prob,[0,0,0]) 
    
    Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]

    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    nor=PtMass/Ptmas

    #normalisation
    dist_bins_om=dist_bins_prob*nor #article distribution function in [N/m] its normalised to each bin width in order to calculate particle number by int(om)

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    print("ptmas"+str(PtMass))
 
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l

#start make distribution
######################################################################
#make a start distribution of particles its normalised first
#then its multiplied by total particle number
#distribution is saved in two vectors vector of bins and vector of particle numbers devided by bin width
#outside of the bins we assume zero particles

#make vectiors from bin parameters
def f_vl_vc(lmin=0.25,lmax=3,cmin=0.25,cmax=8):
    
    vec_c_borders=np.arange(0, cmax, cmin)
    vec_l_borders=np.arange(0, lmax, lmin)

    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    vec_c=np.zeros(Nc)
    vec_l=np.zeros(Nl)

    #create vectors of bin mids. 
    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5

    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9

    return vec_c,vec_l

#make start dist, but bring in cdist
def f_start_dist_cin(cin,lmin1=0.25,lmax1=5.,cmin1=0.25,MPars=ModPars, loc_l=0.25, scale_l=0.25):

    Nr=len(cin)
    vec_r_borders = np.arange(0, (Nr+1) * cmin1, cmin1)
    vec_r=np.zeros(Nr)
    for i in range(Nr):
        vec_r[i] = (vec_r_borders[i]+vec_r_borders[i+1])*0.5
    #Nc=len(cin)
    
    vec_l_borders = np.arange(0, lmax1, lmin1)
    Nl=len(vec_l_borders)-1
    vec_l=np.zeros(Nl)
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5

    Nc=Nr-Nl+1
    vec_c=np.zeros(Nc)
    vec_c_borders = np.arange(0, (Nc+1) * cmin1, cmin1)
    #create vectors of bin mids. 

    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5

    C,L = np.meshgrid(vec_c,vec_l)
    pdf = np.zeros(C.shape)
    
    #dc2 = (vec_c[1] - vec_c[0]) * 0.5
    #dr2 = dc2
    #vec_r= np.arange(dr2,vec_l[-1]+vec_c[-1],2*dr2)
    
    distr_l = norm( loc=loc_l, scale=scale_l ) #parameter for pt shell
    distr_c = cin #parameters for every variable.    

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            #pdf[i,j] = distr.pdf([C[i,j], L[i,j]])
            pdf[i,j] = distr_l.pdf(L[i,j])*distr_c[j] #could be i
            
            if vec_l[i] >  2 - vec_c[j] :
                pass
                #pdf[i,j] = pdf[i,j]*10**-8
            else:
                #pdf[i,j] = pdf[i,j]*10**-16
                pass

    pdf[0,0]=0 #set zero vals in first bin
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    dist_bins_prob=pdf
    #empty the first bin for stability reasons

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    X = f_mat2vec(dist_bins_prob,[0,0,0])
    
    #this it should return pt conc why convert t conc also norm to conc
    
    #Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]
    
    pt_c_targ=ModPars[23]*1e-3/(Pt_mol_mas*ModPars[20]*ModPars[19])
    pt_c_cur=f_stats(X,MPars,vec_c,vec_l)[-2]

    #print(pt_c_cur)
    #print(pt_c_targ)

    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    #nor=PtMass/Ptmas
    nor=pt_c_targ/pt_c_cur

    dist_bins_om=dist_bins_prob*nor

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
 
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    #print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l



#make start dist, but bring in cdist
#changed in order to have cin input, but the original should be working
def f_start_dist_cin_1(cin,lmin1=0.25,lmax1=5.,cmin1=0.25,MPars=ModPars, loc_l=0.25, scale_l=0.25):

    Nr=len(cin)
    vec_r_borders = np.arange(0, (Nr+1) * cmin1, cmin1)
    vec_r=np.zeros(Nr)
    for i in range(Nr):
        vec_r[i] = (vec_r_borders[i]+vec_r_borders[i+1])*0.5
    #Nc=len(cin)
    
    vec_l_borders = np.arange(0, lmax1, lmin1)
    Nl=len(vec_l_borders)-1
    vec_l=np.zeros(Nl)
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5

    Nc=Nr-Nl+1
    vec_c=np.zeros(Nc)
    vec_c_borders = np.arange(0, (Nc+1) * cmin1, cmin1)
    #create vectors of bin mids. 

    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5

    C,L = np.meshgrid(vec_c,vec_l)
    pdf = np.zeros(C.shape)
    
    #dc2 = (vec_c[1] - vec_c[0]) * 0.5
    #dr2 = dc2
    #vec_r= np.arange(dr2,vec_l[-1]+vec_c[-1],2*dr2)
    
    distr_l = norm( loc=loc_l, scale=scale_l ) #parameter for pt shell
    distr_c = cin #parameters for every variable.    

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            #pdf[i,j] = distr.pdf([C[i,j], L[i,j]])
            pdf[i,j] = distr_l.pdf(L[i,j])*distr_c[j] #could be i
            
            if vec_l[i] >  2 - vec_c[j] :
                pass
                #pdf[i,j] = pdf[i,j]*10**-8
            else:
                #pdf[i,j] = pdf[i,j]*10**-16
                pass

    pdf[0,0]=0 #set zero vals in first bin
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    dist_bins_prob=pdf
    #empty the first bin for stability reasons

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    X = f_mat2vec(dist_bins_prob,[0,0,0])
    
    #this it should return pt conc why convert t conc also norm to conc
    
    #Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]
    
    pt_c_targ=ModPars[23]*1e-3/(Pt_mol_mas*ModPars[20]*ModPars[19])
    pt_c_cur=f_stats(X,MPars,vec_c,vec_l)[-2]

    #print(pt_c_cur)
    #print(pt_c_targ)

    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    #nor=PtMass/Ptmas
    nor=pt_c_targ/pt_c_cur

    dist_bins_om=dist_bins_prob*nor

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
 
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    #print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l

### an alternative function to construct the starting particle distribution
### minumal and maximal bin are given, spacing is determined by minimal bin size
### modpars parameters provided in order to normalise to platinum loading
#variables:
# lmin = shell minimal bin
# lmax = shell largest bin
# cmin = core minimal bin
# cmax = core largest bin
# MPars = model parameters 
# loc_l= shell normal distribution mean
# scale_l = shell normal distribuion variance
# s_c = core lognorm distribution s
# loc_c = core lognorm distribution mean 
# scale_c = core lognorm distribution scale
def f_start_dist_alter(lmin=0.25,lmax=5,cmin=0.25,cmax=10,MPars=ModPars, loc_l=0.25, scale_l=0.25, s_c=0.7, loc_c=0.4, scale_c=2):

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    #print(10**-3*NPtTot/VolCat)

    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]

    vec_c_borders=np.arange(0, cmax, cmin)
    vec_l_borders=np.arange(0, lmax, lmin)

    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    #print("Nc")
    #print(Nc)
    #print("Nl")
    #print(Nl)

    vec_c=np.zeros(Nc)
    vec_l=np.zeros(Nl)

    #create vectors of bin mids. 
    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5
        
    C,L = np.meshgrid(vec_c,vec_l)
    # Initializing the covariance matrix

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix

    distr_l = norm(loc=loc_l, scale=scale_l)
    distr_c = lognorm(s_c,loc=loc_c, scale=scale_c)
    #distr_c = norm(loc=4, scale=2)
    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    
    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(C.shape)
    
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            #pdf[i,j] = distr.pdf([C[i,j], L[i,j]])
            pdf[i,j] = distr_l.pdf(L[i,j])*distr_c.pdf(C[i,j])
            
            if vec_l[i] >  2 - vec_c[j] :
                pass
                #pdf[i,j] = pdf[i,j]*10**-8
            else:
                #pdf[i,j] = pdf[i,j]*10**-16
                pass

    pdf[0,0]=0
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    dist_bins_prob=pdf
    #empty the first bin for stability reasons

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    X = f_mat2vec(dist_bins_prob,[0,0,0])
    
    #this it should return pt conc why convert t conc also norm to conc
    
    #Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]
    
    pt_c_targ=ModPars[23]*1e-3/(Pt_mol_mas*ModPars[20]*ModPars[19])
    pt_c_cur=f_stats(X,MPars,vec_c,vec_l)[-2]

    #print(pt_c_cur)
    #print(pt_c_targ)

    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    #nor=PtMass/Ptmas
    nor=pt_c_targ/pt_c_cur

    dist_bins_om=dist_bins_prob*nor

    #test
    #print("new stats")
    #print(f_stats( f_dist_to_X(dist_bins_om),ModPars,vec_c,vec_l)[-2])
    

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
 
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    #print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l


def f_start_dist_alter_norm(lmin=0.25,lmax=5,cmin=0.25,cmax=10,MPars=ModPars, loc_l=0.25, scale_l=0.25, loc_c=0.25, scale_c=0.25):

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    #print(10**-3*NPtTot/VolCat)

    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]

    vec_c_borders=np.arange(0, cmax, cmin)
    vec_l_borders=np.arange(0, lmax, lmin)

    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    #print("Nc")
    #print(Nc)
    #print("Nl")
    #print(Nl)

    vec_c=np.zeros(Nc)
    vec_l=np.zeros(Nl)

    #create vectors of bin mids. 
    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5
        
    C,L = np.meshgrid(vec_c,vec_l)
    # Initializing the covariance matrix

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix

    distr_l = norm(loc=loc_l, scale=scale_l)
    distr_c = norm(loc=loc_c, scale=scale_c)
    #distr_c = norm(loc=4, scale=2)
    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    
    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(C.shape)
    
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            #pdf[i,j] = distr.pdf([C[i,j], L[i,j]])
            pdf[i,j] = distr_l.pdf(L[i,j])*distr_c.pdf(C[i,j])
            
            if vec_l[i] >  2 - vec_c[j] :
                pass
                #pdf[i,j] = pdf[i,j]*10**-8
            else:
                #pdf[i,j] = pdf[i,j]*10**-16
                pass

    pdf[0,0]=0
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    dist_bins_prob=pdf
    #empty the first bin for stability reasons

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    X = f_mat2vec(dist_bins_prob,[0,0,0])
    
    #this it should return pt conc why convert t conc also norm to conc
    
    #Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]
    
    pt_c_targ=ModPars[23]*1e-3/(Pt_mol_mas*ModPars[20]*ModPars[19])
    pt_c_cur=f_stats(X,MPars,vec_c,vec_l)[-2]

    #print(pt_c_cur)
    #print(pt_c_targ)

    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    #nor=PtMass/Ptmas
    nor=pt_c_targ/pt_c_cur

    dist_bins_om=dist_bins_prob*nor

    #test
    #print("new stats")
    #print(f_stats( f_dist_to_X(dist_bins_om),ModPars,vec_c,vec_l)[-2])
    

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
 
    VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    NPtTot=PtMass/molarPt
    #print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l

#a test distribution construcotr that only makes particels in one bin. it doesnt normalise in order to test the nroma function as well.
# returns a distribution containing a single full bin
#variables:
# lmin = shell minimal bin
# lmax = shell largest bin
# cmin = core minimal bin
# cmax = core largest bin
# MPars = model parameters 
# n_parts = anumber of particles
# bin = a bin to populate
def f_start_dist_1bin(lmin=0.25,lmax=5,cmin=0.5,cmax=10,MPars=ModPars,n_parts=0,bin=(5,5),Pt_target=10.):
    
    vec_c_borders=np.arange(0, cmax, cmin)
    vec_l_borders=np.arange(0, lmax, lmin)

    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    #print("Nc")
    #print(Nc)
    #print("Nl")
    #print(Nl)

    vec_c=np.zeros(Nc)
    vec_l=np.zeros(Nl)

    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5
        
    C,L = np.meshgrid(vec_c,vec_l)

    pdf = np.zeros(C.shape)
    
    if n_parts != 0 :
        pdf[bin[0],bin[1]]=n_parts
    else:
        pdf[bin[0],bin[1]]=1.


    #transform to nms
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    #onlly norm to partile number
    
    Dc=vec_c[1]-vec_c[0]
    Dl=vec_l[1]-vec_l[0]
    
    num=Dc*Dl*pdf[bin[0],bin[1]]

    if n_parts != 0 :
        norm=n_parts/num
    else:
        norm=f_norm_pt(pdf ,vec_c, vec_l,MPars,Pt_target)
    
    dist_bins_om=pdf*norm
    
    return dist_bins_om ,vec_c, vec_l
    
#returns max concentration of Pt in cell acording to loading
#returns norm of distribution to quantity of platinum.
#target Ptis in units of concetration
def f_norm_pt(dist_bins_om ,vec_c, vec_l,MPars,Pt_target):
    #print("in norm fuction")
    #print(dist_bins_om)
    X=f_dist_to_X(dist_bins_om)
    #print(X)
    #f_stats2 can also calculate pure platinum distribution
    Pt_c=f_stats(X,MPars,vec_c,vec_l)[-2]
    #Pt_mols=(muNaf*A*L*1e3)*Pt_c
    #Ptmas = molarPt*Pt_mols
    #print(Pt_target)
    #print(Pt_c)
    Pt_target/Pt_c
    return Pt_target/Pt_c
 
   
#create distribution in platinum
#distribution is normalised to avaliable platinum loading
#variables:
# lmin = shell minimal bin
# lmax = shell largest bin
# cmin = core minimal bin
# cmax = core largest bin
# MPars = model parameters 
# loc_l = normal dist center
# scale_l = normal dist spread
def f_start_dist_Pt(lmin=0.25,lmax=5,cmin=0.5,cmax=10,MPars=ModPars, loc_l=2, scale_l=0.65,Pt_target=10.):

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g loading times surface
    
    #print("ptmas"+str(PtMass))
    #VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    #NPtTot=PtMass/molarPt #atomic number of Pt
    #print(10**-3*NPtTot/VolCat)

    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]

    #generate distribution bins
    vec_c_borders=np.arange(0, cmax, cmin)
    vec_l_borders=np.arange(0, lmax, lmin)

    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    #print("Nc")
    #print(Nc)
    #print("Nl")
    #print(Nl)

    vec_c=np.zeros(Nc)
    vec_l=np.zeros(Nl)

    for i in range(Nc):
        vec_c[i] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5
        
    C,L = np.meshgrid(vec_c,vec_l)
    # Initializing the covariance matrix

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix

    distr_l = norm(loc=loc_l, scale=scale_l)
    #removed this since we are only dealing in one bin of platinum
    #distr_c = lognorm(s_c,loc=loc_c, scale=scale_c)
    #distr_c = norm(loc=4, scale=2)
    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    
    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(C.shape)
    
    #seting only for smalest core particles
    j=0
    for i in range(C.shape[0]):
        #for j in range(C.shape[1]):
            #pdf[i,j] = distr.pdf([C[i,j], L[i,j]])
            pdf[i,j] = distr_l.pdf(L[i,j])
            
            if vec_l[i] >  2 - vec_c[j] :
                pass
                #pdf[i,j] = pdf[i,j]*10**-8
            else:
                #pdf[i,j] = pdf[i,j]*10**-16
                pass

    #only death in zero bin
    pdf[0,0]=0
    #correct units to nm
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    
    #dist_bins_prob=pdf
    
    #empty the first bin for stability reasons

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    #X = f_mat2vec(dist_bins_prob,[0,0,0])
    
    #transorm to concentration
    #Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]

    nor=f_norm_pt(pdf ,vec_c, vec_l,MPars,Pt_target)
    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    #nor=PtMass/Ptmas

    dist_bins_om=pdf*nor

    #PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
 
    #VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    #NPtTot=PtMass/molarPt
    #print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l

#change dist to create the distributuion in th enew zero bin
def f_start_dist_Pt1(lmin=0.25,lmax=5,cmin=0.25,cmax=10,MPars=ModPars, loc_l=2, scale_l=0.65,Pt_target=10.):

    PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g loading times surface
    
    #print("ptmas"+str(PtMass))
    #VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    #NPtTot=PtMass/molarPt #atomic number of Pt
    #print(10**-3*NPtTot/VolCat)

    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]

    #generate distribution bins
    vec_c_borders=np.arange(0, cmax, cmin)
    vec_l_borders=np.arange(0, lmax, lmin)

    Nc=len(vec_c_borders)-1
    Nl=len(vec_l_borders)-1

    #print("Nc")
    #print(Nc)
    #print("Nl")
    #print(Nl)

    vec_c=np.zeros(Nc+1)
    vec_l=np.zeros(Nl)

    #first core bin is left for zero core
    for i in range(Nc):
        vec_c[i+1] = (vec_c_borders[i]+vec_c_borders[i+1])*0.5
    for i in range(Nl):
        vec_l[i] = (vec_l_borders[i]+vec_l_borders[i+1])*0.5
        
    C,L = np.meshgrid(vec_c,vec_l)
    # Initializing the covariance matrix

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix

    distr_l = norm(loc=loc_l, scale=scale_l)
    #removed this since we are only dealing in one bin of platinum
    #distr_c = lognorm(s_c,loc=loc_c, scale=scale_c)
    #distr_c = norm(loc=4, scale=2)
    # Generating a meshgrid complacent with
    # the 3-sigma boundary
    
    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(C.shape)
    
    #seting only for smalest core particles
    j=0
    for i in range(C.shape[0]):
        #for j in range(C.shape[1]):
            #pdf[i,j] = distr.pdf([C[i,j], L[i,j]])
            pdf[i,j] = distr_l.pdf(L[i,j])
            
            if vec_l[i] >  2 - vec_c[j] :
                pass
                #pdf[i,j] = pdf[i,j]*10**-8
            else:
                #pdf[i,j] = pdf[i,j]*10**-16
                pass

    #only death in zero bin
    pdf[0,0]=0
    #correct units to nm
    vec_c=vec_c*10**-9
    vec_l=vec_l*10**-9
    
    #dist_bins_prob=pdf

    #empty the first bin for stability reasons

    #normalise particle number to platinum loading.
    #transform 2d distribution to 1D calcualtion vector
    #X = f_mat2vec(dist_bins_prob,[0,0,0])
    
    #transorm to concentration
    #Ptmas = molarPt*(muNaf*A*L*1e3)*f_stats(X,MPars,vec_c,vec_l)[-2]

    nor=f_norm_pt(pdf ,vec_c, vec_l,MPars,Pt_target)
    #normalisation constant. Ratio between distribution Pt mass and nominal Pt mass
    #nor=PtMass/Ptmas

    dist_bins_om=pdf*nor
    #PtMass=ModPars[23]*ModPars[21] # Mass Of platinum in g
    #print("ptmas"+str(PtMass))
 
    #VolCat=ModPars[20]*ModPars[21]*ModPars[19] #cathode volume
    #NPtTot=PtMass/molarPt
    #print("max_C_Pt"+str(10**-3*NPtTot/VolCat))

    #return distribution, bin minds
    return dist_bins_om ,vec_c, vec_l


# returns a normalised distribution 1D
#inputs
# x
# y
def f_norm_dist1d(x,y):
    dx=x[1]-x[0]
    norm=dx*np.sum(y)
    return y/norm

#returns a sum of squares of two vectors
#varibles:
# test = input vector 1
# target =  input vector 2
def f_sum_squares(test,target):
    return np.sum(np.abs(test-target))

#returns sum of absolute values normalised by interval step
#veriables:
#test = vector 1 values
#target = vector 2 values
#x = independant variable
def f_sum_abs(test,target,x):
    dx=x[1]-x[0]
    return dx*np.sum(np.abs(test-target))

#returns a X vector for f_dcdl iteration from 2D PSD 
# dist  = 2D PDS
# vec_ns= [cpt,xO,cCu] vector of concentrations of species
def f_dist_to_X(dist,vec_ns=[0,0,0]):   
    nSt = len(vec_ns)
    #number of bins
    nc,nl=np.shape(dist)
    # total number of variables
    nSys = nSt+nc*nl

    ### set initial distribution
    #initial states
    X = np.zeros(nSys)

    #fill in the vriables
    for i in range(nSt):
        X[i]=vec_ns[i]
    
    #fill in the distribution
    X[nSt:]=dist.flatten()
    
    return X

#returns the RHS derivatives for electrochemistry and PSD
#it returns dX,dom,Dc,Dl. usefull for debuging. 
#variables:
# t = time
# X = state vector
# ModPars = parameters vector
# dist_cs = core_bins
# dist_ls = shell bins
# SFT = temperature(t)
# SFPt = Pt potential
# SFH = pH(t)
@jit(nopython=True,parallel=isparalel)
def f_dc_dl(t,X,ModPars,dist_cs,dist_ls,SFT,SFPt,SFH):
    
    #print("t="+"{:.2e}".format(t))
    lenX=len(X)    
    nSt=3
    
    #nAll=1 #to generalize in 1D case
    #nBins=len(dist_bins_mids) #number of bins
    nc=len(dist_cs)
    nl=len(dist_ls)
    
    #nSt=int(len(X)/nAll) # number of variables
    #print("lenzdat="+str(nAll))
    #print("lennSt="+str(nSt))

    #read the common variables
    CPt=X[0] #pt concentr  ation
    fiPtO=X[1] #oxide coverage
    #CCu=X[2] #cu concentration #dont need it in calculation
    
    # fuel cell states; these are inputs as functions of t
    T = SFT(t) #input oxigen concentration bot needed because reaction is with water
    cHpin = SFH(t) #input proton concentration
    UPtin = SFPt(t) #inpput voltage on Pt particle (from ionomer)    
    #pH=-np.log10(cHpin) #calculate pH for further calculation. Curently not used
    
    #vector of derivatives to be filled
    dX = np.zeros(lenX)

    #input parameters transformed for easier calculation
    nuForPtO    =   ModPars[0]  #Forward Pt oxide rate [1/s]
    nuBacPtO    =   ModPars[1]  #backward Pt oxide rate [1/s]
    #GamaPt      =   ModPars[2]  #Pt surface site density [mol/m2]
    HEnthPtO    =   ModPars[3]  #Partial molar oxide formation enthalpy (0 coverage) [J/mol]
    betPtO      =   ModPars[4]  #Butler-Volmer transfer coefficient; PtO formation []
    nElPtO      =   ModPars[5]  #electrons transfered during Pt oxidation []
    UPtO        =   ModPars[6]  #bulk PtO eq voltage [V]
    lamPtO      =   ModPars[7]  #PtO dependant kinetic barier constant [J/mol]
    omegaPto    =   ModPars[8]  #PtO-PtO interaction energy [J/mol]
    
    nuForPt     =   ModPars[9]  #Dissolution atempt frequency [1/s]
    nuBacPt     =   ModPars[10] #back dissolution rate [1/s]
    HEnthPt     =   ModPars[11] #Partial molar formation enthalpy (0 coverage) [J/mol]
    betPt       =   ModPars[12] #Butler-Volmer transfer coefficient Pt []
    nElPt       =   ModPars[13] #Pt dissolution electron transfer []
    UPt         =   ModPars[14] #Equelibrium voltage Pt dissolution [V]
    
    #OmegaPt     =   ModPars[15] #Ptmolar volume [m3/mol]
    gamma       =   ModPars[16] #Pt [111] surface tension [J/m2]
    CPtRef      =   ModPars[17] #reference Pt conc
    xiH2O       =   ModPars[18] #volume fraction of water in ionomer #not used but it could be implemented alongside muNaf
    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]
    eps_u       =   ModPars[22] #eps_U ligand effecto correction factor 
    #PtLoad      =   ModPars[23] #Pt loading [g/m2]   
    #OmegaCu     =   ModPars[24] #Cumolar volume [m3/mol]    
    n           =   ModPars[25] #Pt/Co ratio  
    
    DifPt = 4*10**-9 #diffusion coeficient platinum. Added in order to smothen numeric behaveour[m2/s]
    ML1 = 0.223*10**-9 #Pt monolayer thickness. under this shell we start 
    
    #start of derivative calculations
    ddc=dist_cs[2]-dist_cs[1]
    ddl=dist_ls[2]-dist_ls[1]

    #print(ddc)
    #print(ddl)
    #create matrices to bi filled 
    
    #c velocity
    Dc=np.zeros((nl,nc))
    #l velocity
    Dl=np.zeros((nl,nc))
    #dXX matrix
    XX=np.zeros((nl,nc))
    ### fill the XX matrix (more convinient) with  X input vector. easier to loop afterwards
    for i in range(nl): 
        XX[i,:]=X[3+nc*i:3+nc*(i+1)]
        
    #XX=np.reshape(X[3:],(nl,nc))
    #loop trough all the bins and caluclate rate in oxide growth and platinum dissolution
    for i,l in enumerate(dist_ls):
        for j,c in enumerate(dist_cs):
            #radius = core + shell
            if c>0.:
                r=c+l
            else:
                r=l
            ### extended gamma parameter here omited for faster calculation
            #gammaTot = gamma + GamaPt*fiPtO*R*T *(\
            #                np.log(nuBacPtO/nuForPtO)   +   \
            #                np.log(cHpin**2)         +   \
            #                nElPt*F**(UPtO-UPtin)/(R*T) +   \
            #                omegaPto*fiPtO/(2*R*T)      +   \
            #                np.log(fiPtO/2)             +   \
            #                ((2-fiPtO)/fiPtO)*np.log(1-fiPtO/2)
            #                )        
            gammaTot=gamma    
            #rates of change Pt
            
            #case one large shells or pure platinum
            

            if r>=6*ML1 or c<=0.:
                
                dina_fact_0=1.
                #dina_fact_0=( 1 + (nuForPt*GamaPt*r/(CPtRef*DifPt))   *np.exp(    (nElPt*F*(betPt)/(R*T))     *(UPt-UPtin-4*OmegaPt*gammaTot/(2*r*nElPt*F))  )  )

                rPt    = nuForPt*GamaPt    *np.exp(-HEnthPt/(R*T)) * (1.-min(1.,fiPtO))     *(\
                        1.                                             *np.exp(    -(nElPt*F*(1-betPt)/(R*T))  *(UPt-UPtin-4*OmegaPt*gammaTot/(2*r*nElPt*F))  )   \
                        -(nuBacPt/nuForPt)*(CPt/CPtRef)               *np.exp(    (nElPt*F*(betPt)/(R*T))     *(UPt-UPtin-4*OmegaPt*gammaTot/(2*r*nElPt*F))  )    \
                        ) / dina_fact_0
            
            #smaller shells includs an epsilon correction for shell thickness
            else:
                #factor to smoothe fnction behaviour at cyclnig
                dina_fact_1=1.
                #dina_fact_1=( 1 + (nuForPt*GamaPt*r/(CPtRef*DifPt))   *np.exp(    (nElPt*F*(betPt)/(R*T))     *(UPt-UPtin-eps_u*l-4*OmegaPt*gammaTot/(2*r*nElPt*F))  )  )
                
                rPt    = nuForPt*GamaPt    *np.exp(-HEnthPt/(R*T)) * (1.-min(1.,fiPtO))     *(\
                        1.                                             *np.exp(    -(nElPt*F*(1-betPt)/(R*T))  *(UPt-UPtin-eps_u*(l-6*ML1)-4*OmegaPt*gammaTot/(2*r*nElPt*F))  )   \
                        -(nuBacPt/nuForPt)*(CPt/CPtRef)               *np.exp(    (nElPt*F*(betPt)/(R*T))     *(UPt-UPtin-eps_u*(l-6*ML1)-4*OmegaPt*gammaTot/(2*r*nElPt*F))  )    \
                        ) / dina_fact_1

            #rate of Cu dissolution #check this
            #this was changed co there should be faster core dissolution
            rCu=rPt*OmegaPt/(OmegaCu + n*OmegaPt)
            
            #derive dR/dt = [dl/dt,dc/dt] change in core and shell diameters.
            #ML1 is critical shell thickness at which we start loosing copper.
            if c>=ML1: #larger core
                if l<=ML1: #first row L1.
                    if -rPt < 0: #negative core growth travel to 0,0 #changes here increased core change due to instantanious coper dissolution here we could say something about even faster disolution 
                        Dc[i,j] = -(rPt*OmegaPt + rCu*OmegaCu) #check if we need to multiply by 1/2.
                        Dl[i,j] = 0
                    else: #positive growth but of shell only #actually this is a bit under question. It is assumed that shell growth procedes normally forom etched sample
                        Dc[i,j]=0
                        Dl[i,j]=-rPt*OmegaPt    
                else: #shell growth only and disolution
                    Dc[i,j]=0
                    Dl[i,j]=-rPt*OmegaPt
            else: # small core
                if l>=ML1: #larger shell
                    Dc[i,j] = 0 #its protected
                    Dl[i,j] = -rPt*OmegaPt #normal disolution or growth of shell
                else:   #small shell and core
                    Dc[i,j] = 0
                    Dl[i,j] = 0


    #oxide coverage rate 
    #treated as a scalar variable without r dependance in order to simplify calculation
    rfiPtO  = nuForPtO*GamaPt   *np.exp(-(HEnthPtO+lamPtO*fiPtO)/(R*T)  )   *(\
        (1.-fiPtO/2.)                                      *np.exp(    -(nElPtO*F*(1-betPtO)/(R*T))    *(UPtO-UPtin+omegaPto*fiPtO/(nElPtO*F)) )   \
        -(nuBacPtO/nuForPtO)*(cHpin**2)*(0.5*fiPtO)        *np.exp(    (nElPtO*F*(betPtO)/(R*T))       *(UPtO-UPtin+omegaPto*fiPtO/(nElPtO*F)) )    \
        )
           
    dfiPtO=rfiPtO/GamaPt
    dX[1]=dfiPtO
    #dX[1]=0
    
    #dN/dt derived from d(N*R_dot)/dR discretisation
    dom=np.zeros((nl,nc))
    
    #absolute values of R_dot R_dot
    a_l=np.abs(Dl)
    a_c=np.abs(Dc)
    
    #heaviside functions. 
    #they are used in upwind discretisation
    #shell
    hs_p_Dl=np.zeros((nl,nc))
    hs_n_Dl=np.zeros((nl,nc))
    #core
    hs_p_Dc=np.zeros((nl,nc))
    hs_n_Dc=np.zeros((nl,nc))

    #construct heaviside vectors
    for j in range(len(dist_cs)):
        for i in range(len(dist_ls)):
            #heaviside v smeri shellov
            hs_p_Dl[i,j]=f_nb_heaviside(Dl[i,j])
            hs_n_Dl[i,j]=f_nb_heaviside(-1*Dl[i,j])  
    
            #heaviside v smeri jeder
            hs_p_Dc[i,j]=f_nb_heaviside(Dc[i,j])
            hs_n_Dc[i,j]=f_nb_heaviside(-1*Dc[i,j])
    
    ####################
    #j direction derivatives
    #most of derivatives are in shell direction only  since this is wherer Pt is deposited or etched
    #loop trough cores (j direction)
    for j in range(len(dist_cs)):
        #loop trough all but the first and the last bin
        #chell derivatives only (in i direction)
        for i in range(1,len(dist_ls)-1,1): 
            dom[i,j]=(1./ddl)*( XX[i-1,j]*a_l[i-1,j]*hs_p_Dl[i-1,j] - XX[i,j]*a_l[i,j] + XX[i+1,j]*a_l[i+1,j]*hs_n_Dl[i+1,j])
        
        #handle first and the last bin. in the first (i=0) we take care there is no back flow
        #in case of growth  -XX[i,j]*a_l[i,j]*hs_p_Dl[i,j]
        #in case of dissolution: XX[i+1,j]*a_l[i+1,j]*hs_n_Dl[i+1,j] 
        i=0 
        dom[i,j]=(1./ddl)*( -XX[i,j]*a_l[i,j]*hs_p_Dl[i,j]  + XX[i+1,j]*a_l[i+1,j]*hs_n_Dl[i+1,j] )
        
        #last bin is handeled in equivalent manner. In the end it doesnt matter much due to 
        i=len(dist_ls)-1
        dom[i,j]=(1./ddl)*( XX[i-1,j]*a_l[i-1,j]*hs_p_Dl[i-1,j] - XX[i,j]*a_l[i,j]*hs_n_Dl[i,j] )
    
    #handling of the seccond smallest bin done separately
    dom[1,0]=(1./ddl)*( -XX[1,0]*a_l[1,0] + XX[2,0]*a_l[2,0]*hs_n_Dl[2,0] )
    
    #####################
    #core derivatives
    #0 for vas majority of particles, since most are protected by shell
    #all the action in the zeroth row ( i = 0 thin shell particles)
    #fluxes are added to existing i direction fluxes (due to divergence of N*R_dot being a sum of two orthogonal therms
    for j in  range(2,len(dist_cs)-1,1):   #first row without edges.
        dom[0,j]=dom[0,j]+(1./ddc)*( XX[0,j-1]*a_c[0,j-1]*hs_p_Dc[0,j-1] - XX[0,j]*a_c[0,j] + XX[0,j+1]*a_c[0,j+1]*hs_n_Dc[0,j+1])

    #larger particles on the far right. Zero flux from the boundary set. It doesnt matter due to low particle number.
    j=len(dist_cs)-1
    dom[0,j]=dom[0,j]+(1./ddc)*( - XX[0,j]*a_c[0,j] )
    
    #flux in particel disolution direction only
    j=1
    dom[0,j]=dom[0,j]+(1./ddc)*( - XX[0,j]*a_c[0,j] + XX[0,j+1]*a_c[0,j+1]*hs_n_Dc[0,j+1])
    
    #hardcoded n change in zeroth bin. In pratice these particles are below critical radius and would disolve instantaniously.
    dom[0,0]=0

    #if we are dealing with combined ditribution, fix the seccond bin
    if dist_cs[0]<=0:
        #print("c=0")
        dom[0,1]=0
        j=2
        dom[0,j]=dom[0,j]+(1./ddc)*( - XX[0,j]*a_c[0,j] + XX[0,j+1]*a_c[0,j+1]*hs_n_Dc[0,j+1])

    j=len(dist_cs)-1 #right corner - large particles
    dom[0,j]=(1./ddc)*( XX[0,j-1]*a_c[0,j-1]*hs_p_Dc[0,j-1] - XX[0,j-1]*a_c[0,j-1]*hs_n_Dc[0,j-1] )
    
    #change from the convenient matrix form.
    dX[3:]=dom.flatten()

    ### calculate change in solute concentrations due to net dissolution of particles
    n           =   ModPars[25] #Pt/Cu
    muNaf       =   ModPars[19] #volume fraction of ionomer in cathode electrode
    L           =   ModPars[20] #CCL thickness [m]
    A           =   ModPars[21] #cell surface area [m^2]

    #OmegaPt     =   ModPars[15]
    #OmegaCu     =   ModPars[24]

    c_tot=0
    l_tot=0

    #lop trough core and shell list
    for j,c in enumerate(dist_cs):
        for i,l in enumerate(dist_ls):
            r = c + l
            num = ddc*ddl*dom[i,j] #dcc in dll sta samo konstanten faktor za vse bine. Ne smeta met veze

            v_core  = num * (4./3.) * 3.14 * c**3
            v_shell = num * (4./3.) * 3.14 * r**3 - v_core

            c_tot=c_tot+v_core
            l_tot=l_tot+v_shell

    #Pt and Cu lost in moles
    Pt = -1*( (l_tot/OmegaPt) + (c_tot/OmegaCu) * (n/(n+1.)) )
    Cu = -1*( (c_tot/OmegaCu) * (1./(n+1.)) )

    #transformed into concentrations.
    dX[0]=Pt*10**-3/(L*A*muNaf)
    dX[2]=Cu*10**-3/(L*A*muNaf)

    return dX,dom,Dc,Dl

#returns the RHS derivatives
#it is a wraper function for f_dcdl
#it only returns the first output of f_dcdl 
@jit(nopython=True,parallel=isparalel)
def f_dfdt(t,X,ModPars,dist_cs,dist_ls,SF1,SF2,SF3):
    #print("t:"+str(t))
    rez=f_dc_dl(t,X,ModPars,dist_cs,dist_ls,SF1,SF2,SF3)
    return rez[0]


#first iteration of cycle skiping function. skips every other step.
#ncyc should be even
#should return data in the same format as solve, so it can be ploted and

#transforms c/l distribution to r for fitting

#function to integrate
#have to check how it works

#transform 1d vector to 2D suitable for plot.
#variables:
def f_trans_x_to_om(X,NS,vec_c,vec_l):
    return np.reshape(X[NS:],(len(vec_l),len(vec_c)))

#calculate Pt to alow wt from molar ratio
def f_mol_from_weight(weight_ratio,):
    
    return mol_ratio

#calculate weiht ratio from 
def f_weight_from_mol():
    
    return weight_ratio


#might take a while maybe i need to optimise this.
#makes a l(r) 2D distribution as well as a new r vector fr the distribution
def f_cl_to_rl_rc(X,NS,vec_l,vec_c):

    res_om=f_trans_x_to_om(X,NS,vec_c,vec_l)

    #l and c half intevals
    dl2 = (vec_l[1] - vec_l[0]) * 0.5 
    dc2 = (vec_c[1] - vec_c[0]) * 0.5

    #does it have to be that way? would it work with any dr?
    dr2 = dc2
    #dr vector
    vec_r=np.arange(dr2,vec_l[-1]+vec_c[-1],2*dr2)

    #r distribution
    #om_r=vec_r.copy()
    #dr half interval
    dr2 = (vec_r[1] - vec_r[0]) * 0.5
    #dr full interval
    #dr = (vec_r[1] - vec_r[0])

    #define a function to integrate in the desired interval diagnoaly
    def f_t_fun(l,c,li,ci):
        if (ci-dc2)<=c<(ci+dc2) and (li-dl2)<=l<(li+dl2):
            return 1.
        else:
            return 0.

    #choose r binning
    lr_mat=np.zeros((len(vec_l),len(vec_r)))
    r_mat=np.zeros((1,len(vec_r)))
    #loop trough r bins
    for k,r in enumerate(vec_r):
        #choose dr interval boundaries
        rmin=r-dr2
        rmax=r+dr2
        #rez=0
        #print(str(k))
        #print(str(r)+" "+str(rmin)+" "+str(rmax))
        #for each r interval loop trough all the c and l intervals
        for i,ci in enumerate(vec_c):
            for j,li in enumerate(vec_l):
                #only go on integrating if we have overlap
                if (li+dl2) > rmin-(ci+dc2) and (li-dl2) < rmax-(ci-dc2):
                    trez=dblquad(f_t_fun, ci-dc2, ci+dc2, lambda x : rmin-x, lambda x : rmax-x, args=(li,ci), epsabs=10e-25, epsrel=10e-4)
                    #if trez[0]<1e-11:
                    #    print(trez[0])
                    #trez=dblquad(t_fun, vec_c[0]-dc2, vec_c[-1]+dc2, rmin, rmax, args=(li,ci), epsabs=10e-22, epsrel=10e-4)               
                    #rez=rez+trez[0]*res_om[j,i,t]
                    #t_mat[j,i]=t_mat[j,i]+trez[0]
                    #had to fix this normalisation!
                    lr_mat[j,k]=lr_mat[j,k]+trez[0]*res_om[j,i]/(2*dl2*2*dr2)
                else:
                    pass
                
    return lr_mat,vec_r

#returns a distribution in r (but we loose l data)
#is this normalised? must check.
#waay faster then above
def f_om_to_r(om):
    # Get the dimensions of the distribution
    nx, ny = om.shape

    # Create a new array to hold the sum distribution
    # The range of sums will be from 0 to nx + ny - 2 (for discrete indices)
    P_z = np.zeros(nx + ny - 1)

    # Iterate through all pairs of (x, y) and add to the appropriate sum bin
    for x in range(nx):
        for y in range(ny):
            z = x + y
            P_z[z] += om[x, y]
    return P_z    

#max r of distribution. usefull for  limitin function execution
def f_r_max(X,vec_l,vec_c):
    om=f_X_to_dist(X,vec_l,vec_c)
    r=f_om_to_r(om)
    #l,r=f_flat_res_om(res,vec_l,vec_r)  #sum to r distribution
    #dl2 = (vec_l[1] - vec_l[0]) * 0.5 
    dc2 = (vec_c[1] - vec_c[0]) * 0.5
    #does it have to be that way? would it work with any dr?
    dr2 = dc2
    #dr vector
    vec_r= np.arange(dr2,vec_l[-1]+vec_c[-1],2*dr2)
    
    argmax = np.argmax(r)
    r_max=vec_r[argmax]
    return r_max


#num = ddc*ddl*dom[i,j] #dcc in dll sta samo konstanten faktor za vse bine. Ne smeta met veze
#function to translate particle of certain core shell to pt and cu concentration

def f_lc_to_ptcu(l,c):
    #OmegaPt     =   ModPars[15] #Ptmolar volume [m3/mol]
    #OmegaCu     =   ModPars[24] #Cumolar volume [m3/mol]    
    n           =   ModPars[25] #Pt/Co ratio  

    r = c + l
    v_core  =  (4./3.) * 3.14 * c**3
    v_shell =  ( (4./3.) * 3.14 * r**3 ) - v_core

    #Pt and Cu lost in moles
    Pt = ( (v_shell/OmegaPt) + (v_core/OmegaCu) * (n/(n+1.)) )
    Cu = ( (v_core/OmegaCu) * (1./(n+1.)) )

    return Pt,Cu

#loop trugh distribution in order to map pt(cu content within each bin.
#variables:
# om = distribution 2D array
# vec_l = shell bins
# vec_c = core bins
#returns:
# omPt = distribution of Pt 
# omCu = distribution of Cu

def f_om_to_ptcu(om,vec_l,vec_c):
    omPt=np.zeros(np.shape(om))
    omCu=np.zeros(np.shape(om))

    ddc=vec_c[1]-vec_c[0]
    ddl=vec_l[1]-vec_l[0]

    for j,c in enumerate(vec_c):
        for i,l in enumerate(vec_l):
            
            Pt,Cu=f_lc_to_ptcu(l,c)
            num = ddc*ddl*om[i,j] 

            omPt[i,j]=Pt*num
            omCu[i,j]=Cu*num

    return omPt, omCu

#flaten the distribution along both axes
#returns distribution in shells and bins
#variables = 
#om = distribution 2D array
#vec_l = shell bins
#vec_c = core bins
def f_flat_res_om(om,vec_l,vec_c):
    #vec l and c only needed for proper normalisation as dimention is decreased
    dl2 = (vec_l[1] - vec_l[0]) * 0.5
    dc2 = (vec_c[1] - vec_c[0]) * 0.5
    #rt_mat=np.sum(crt_mat, axis=0)
    lt_mat=np.sum(om, axis=1)*dc2*2
    ct_mat=np.sum(om, axis=0)*dl2*2

    return lt_mat,ct_mat

#calucalte square diff from target distribution
#data must be numpy arrays
#variables:
# data0
# data1

def f_calc_square(data0,data1):
    return (data0-data1)**2

#calculate delta X for two distributions
#used in order to quantify chenges in particle numbers
#variables:
# X0 = distribution 1 
# X1 = distribution 2
# ModPars = parameters vector
# dist_cs = core bins
# dist_ls = shell bins
def f_calc_delta(X0,X1,ModPars,dist_cs,dist_ls):
    #difference in distribution
    dX=X1-X0 
    #calculate stats of delta of the distributions in roder to get change in Pt and Cu in particles.
    stats=f_stats(dX,ModPars,dist_cs,dist_ls)
    #pt change in pt content
    dX[0]=-stats[-2]
    # we dont calculate change in oxide coverage since it is no relevant
    dX[1]=0
    #Cu change in coper content
    dX[2]=-stats[-1]
    return dX

#solve an AST degradation.
#variables:
# Ncyc,
# Tcyc,
# X0,
# pars,
# method=0,
# startCyc=20,
# tol_r=1e-5,
# tol_a=1e-9

def f_cyc_sim(Ncyc,Tcyc,X0,pars,method=0,startCyc=20,tol_r=1e-5,tol_a=1e-9):
    tMax=1*Tcyc #recalculate the final time
    #tMax=1000
    iDim = 100-1  #number of observation points at which we save data
    dt = tMax/(iDim)
    ti = np.arange(0,tMax+dt,dt)

    #starting time and result. to be apended to.
    rez = np.reshape(X0, (-1, 1))
    t=[0]

    #full calf for first n cycles. We assume, its relevant
    icyc = 0

    #while we havent finished
    while icyc < Ncyc:
        print("calculating start cyc "+str(icyc))
        #solve starting cycles
        if icyc < startCyc:
            Xs=rez[:,-1]
            tStart = time.time()
            sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), Xs, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a) #calculate a cycle 
            tEnd = time.time()
            print("tCalc= "+str(tEnd-tStart))
            rez=np.hstack((rez,np.reshape(sol0.y[:,-1], (-1, 1)))) #append last to rez
            icyc += 1
            t.append((icyc)*Tcyc)
        
        #solve non starting cycle, possible cycle skiping
        else:
            #no skipping
            if method == 0:
                print("calculating cyc "+str(icyc))
                Xs=rez[:,-1]
                tStart = time.time()
                sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), Xs, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a) #calculate a cycle 
                tEnd = time.time()
                print("tCalc= "+str(tEnd-tStart))
                rez=np.hstack((rez,np.reshape(sol0.y[:,-1], (-1, 1)))) #append last to rez
                icyc += 1
                t.append((icyc)*Tcyc)
            #simple dual skip
            elif method == 1:
                print("calculating adaptive method "+str(method)+" cyc "+str(icyc))
                #make a step and calculate delta
                Xs=rez[:,-1]
                tStart = time.time()
                sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), Xs, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a) #calculate a cycle 
                tEnd = time.time()
                print("tCalc= "+str(tEnd-tStart))
                rez=np.hstack((rez,np.reshape(sol0.y[:,-1], (-1, 1)))) #append last to rez
                icyc += 1
                t.append((icyc)*Tcyc)

                #skip a step
                print("skiping cyc "+str(icyc))
                dX=f_calc_delta(Xs,rez[:,-1],pars)
                print(dX[:3])
                X = rez[:,-1] + dX
                rez=np.hstack((rez,np.reshape(X, (-1, 1)))) #append last to rez
                icyc += 1
                t.append((icyc)*Tcyc)

            #progressive exponential time skiping
            elif method == 2: 
                print("calculating adaptive method "+str(method)+" cyc "+str(icyc))

                #make a step and calculate delta
                Xs=rez[:,-1]
                tStart = time.time()
                sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), Xs, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a) #calculate a cycle 
                tEnd = time.time()
                print("tCalc= "+str(tEnd-tStart))
                rez=np.hstack((rez,np.reshape(sol0.y[:,-1], (-1, 1)))) #append last to rez
                icyc += 1
                t.append((icyc)*Tcyc)

                #skip some stepes
                cn=3
                print("calculating adaptive method "+str(method)+" cyc "+str(icyc)+ ". skipping "+str(cn)+" steps.")
                dX=f_calc_delta(Xs,rez[:,-1],pars)
                print(dX[:3])
                X = rez[:,-1] + dX*cn
                rez=np.hstack((rez,np.reshape(X, (-1, 1)))) #append last to rez
                icyc += cn
                t.append((icyc)*Tcyc)
                
    return rez,t

#a function to skip cycles
#variables:
#X0 starting state vector
#Tcyc not used
#ncyc = number of cycles

def f_skip_cyc(X0,Tcyc,ncyc=6):
    t0=1200 #cycle duration in  [s]
    #t0=data[-1,-1]
    p=2*t0
    nCyc = 4 #Number of cycles
    tMax=nCyc*t0 #recalculate the final time
    #tMax=1000
    iDim = 400-1  #number of observation points at which we save data
    dt = tMax/(iDim)
    ti = np.arange(0,tMax+dt,dt)

    #first (few) iterations
    tStart = time.time()
    sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), X0, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)
    tEnd = time.time()
    print("tCalc= "+str(tEnd-tStart))

    for i in range(ncyc):
    #this format for the main function as well

    #get delta:
        #starting X
        X0
        #X after one cycle
        sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), X0, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)
        #delta X and Concentrations. #if bin would empty, set to default low value or smth...

    return rez

#TODO check:
pars=None

#12.2.25 cycling funcitons
def f_run_cyc(Xin,t0,T0=0,ncyc=6,pars_in=pars, tol_r=1e-6,tol_a=1e-6,):
    #sprint("caluculting cycles: "+str(ncyc))
    #create t interval basicaly T interval is connected with cycle state, could be also done differently
    iDim = 400-1  #number of observation points at which we save data
    #t0=Am/(rateMax)
    tMax=ncyc*t0 + T0 #recalculate the final time
    #iDim=10
    #dt = tMax/(iDim)
    #print("T0, TMax")
    #print(T0)
    #print(tMax)

    #ti = np.arange(T0,tMax+dt,dt)
    ti = np.linspace(T0,tMax,iDim)

    #print("start stop times:")
    #print(ti[0])
    #print(ti[-1])

    #first (few) iterations
    #tStart = time.time()
    sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), Xin, method='LSODA', t_eval=ti, args=pars_in, rtol  = tol_r, atol = tol_a)
    #tEnd = time.time()
    #print("tCalc= "+str(tEnd-tStart))

    #sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), Xin, method='LSODA', t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)
    #delta X and Concentrations. #if bin would empty, set to default low value or smth...
    #Xin=sol0.y[:,-1]
    return sol0

def f_calc_delta(X0,X1,ModPars,dist_cs,dist_ls):
    #difference in distribution
    dX=X1-X0 
    #calculate stats of delta of the distributions in roder to get change in Pt and Cu in particles.
    stats=f_stats(dX,ModPars,dist_cs,dist_ls)
    #pt change in pt content
    dX[0]=-stats[-2]
    # we dont calculate change in oxide coverage since it is no relevant
    dX[1]=0
    #Cu change in coper content
    dX[2]=-stats[-1]
    return dX



#skips a bunch of cycles
#TODO: nested execution failsafes
#TODO: breaks

def f_test_cycle_skip(MC=100,round_factor=0.8, round_max_f=0.2,pars=pars,X0=None):

    #target
    #
    
    tStart = time.time()

    #start cycles
    s_cycs=2
    #all positive bins never happens in this cycling
    d_cyc=2
    #finish steps
    finish_steps=3
    
    t0=0
    rez_stat_4=f_stats(X0,ModPars,vec_c,vec_l)
    rez_stat_4=np.array(rez_stat_4)
    rez_t_4=[t0]
    Xin=X0
    rez_X_4=np.array(X0)
    rez_n_4=[0]
    
    print(np.shape(rez_n_4))
    print(np.shape(rez_stat_4))
    print(np.shape(rez_X_4))

    round_max = int(round_max_f*MC)
    print("round_max "+str(round_max))
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    base_filename = f"data_"+str(round_max)+f"_sim_{current_date}.txt"
    filename =  f_create_unique_filename(base_filename)
    #f_append_to_file(filename,str(0) + " " + str(X0) )
    
    i=0
    while i<MC:
        #print(np.shape(rez_n_4))
        #print(np.shape(rez_stat_4))
        #print(np.shape(rez_X_4))
        print(i)
        #initialise maybe calc a few cycles
        #this remains the same as previous versions
        if i<s_cycs:
            print("mode 0: calculating "+str(s_cycs)+" cycles")
            #sol=f_run_cyc(Xin,0,s_cycs)
            #sol=updated_pt_aloy_fit_all_2.f_run_cyc(Xin,Am/(rateMax),0,s_cycs)
            sol=f_run_cyc(Xin,Am/(rateMax),0,s_cycs,pars,tol_r,tol_a)

            Xin=sol.y[:,-1]
            t0=sol.t[-1]
    
            trez=Xin
            trez_stat=f_stats(trez,ModPars,vec_c,vec_l)
            rez_stat_4=np.vstack((rez_stat_4,trez_stat))
            #rez_t_2.append(t0)
    
            rez_X_4=np.vstack((rez_X_4,Xin))
            
            i=i+s_cycs
            rez_n_4.append(i)
            
        else:
            print("mode 1: calculating "+str(1)+" cycle")
            #calc one cycle to get dX
            X01=Xin
            #run a cycle and calculate dX
            #sol=updated_pt_aloy_fit_all_2.f_run_cyc(Xin,Am/(rateMax),0,1)
            tStart = time.time()
            sol=f_run_cyc(Xin,Am/(rateMax),0,1,pars,tol_r,tol_a)
            tEnd = time.time()
            print("Dt= "+str(tEnd-tStart))

            Xin=sol.y[:,-1]
    
            if np.any(Xin[4:])<=0:
                print("encountered negative value")
                break
            
            dX=f_calc_delta(X01,Xin,ModPars,vec_c,vec_l)
            
            i=i+1
            #this may need to be changed in ver 2 in order to fix indices
            Xin=sol.y[:,-1]
            #we increment this anyways
    
            #check how much iteration we can afford. 
            #loop trough all the ratios in diferentials.
            #increment dX
            #print(min(abs(Xin)))
            #print(min(abs(dX)))
            Xratios=Xin/dX
            
            #check wat is the smallest negative value and also report index.
            max_ratio= np.max(Xratios[Xratios < 0]) if np.any(Xratios < 0) else None
            
            print("max_ratio= "+str(max_ratio))
            #treshold = 2.
    
            #all positive for some reason
            #not very likely
            if max_ratio == None:
                print("\tIF = all positive: keep going")
                #skip d_cycs
                Xin=Xin+dX*d_cyc
                i=i+d_cyc
    
            #we can actually skip a few steps
            elif abs(max_ratio) > 2:
                rounded_down = max_ratio
                print("rounded down "+str(rounded_down))
                rounded_down = round_factor
                print("rounded down "+str(rounded_down))
                rounded_down = max_ratio*round_factor
                print("rounded down "+str(rounded_down))
                rounded_down = abs(max_ratio*round_factor)
                print("rounded down "+str(rounded_down))
                rounded_down = int(abs(max_ratio*round_factor))
                print("rounded down "+str(rounded_down))
                rounded_down = min( rounded_down ,round_max)
                print("rounded down "+str(rounded_down))
                #rounded_down=max(rounded_down,1)
                print("rounded down "+str(rounded_down))
                #check if we can finish get close and do a few loops
                #do some loops
                #finish up
                #break
                if i+rounded_down >= MC:
                    print("finishing")
                    #first do some steps then finish up
                    if MC-i > finish_steps:
                        d_cyc=MC-i-finish_steps
                        Xin=Xin+dX*d_cyc  
                        sol=f_run_cyc(Xin,Am/(rateMax),0,finish_steps,pars,tol_r,tol_a) 
                        i=MC
                    #just finish up
                    else:
                        sol=f_run_cyc(Xin,Am/(rateMax),0,finish_steps,pars,tol_r,tol_a) 
                        i=MC
                    break
                
                print("\tIF > 2: cycle "+ str(rounded_down))
                Xin=Xin+dX*rounded_down
                i=i+rounded_down
    
            #we cant propagate
            else:
                print("\tIF < 2: repeate")
                #we have only calculated 
                pass
    
            #record vector and stats and index values at the end.
            #f_append_to_file(filename,str(i) + " " + str(Xin) )
            
    rez_X_4=np.vstack((rez_X_4,Xin))
    trez_stat=f_stats(Xin,ModPars,vec_c,vec_l)
    rez_stat_4=np.vstack((rez_stat_4,trez_stat))
    rez_n_4.append(i)

    w_string=str(i) + " " + np.array2string(Xin, threshold=np.inf, precision=8, separator=", ").replace("\n", "")
    f_append_to_file(filename,w_string)
    tEnd = time.time()
    f_append_to_file(filename,str(tEnd-tStart))

    print("finish:" + str(i))
    #return rez_X_4,rez_stat_4,rez_n_4
    return sol

#helper functions to handle file loging during dif evo.
#function to create a unique fname if the one we requre is taken
#variables:
#base_filename = filename
def f_create_unique_filename(base_filename):
    """Create a unique filename by appending a number if the file already exists."""
    if not os.path.exists(base_filename):
        return base_filename

    # Extract the file base name and extension
    file_base, file_extension = os.path.splitext(base_filename)
    i = 1

    while True:
        new_filename = f"{file_base}_{i}{file_extension}"
        if not os.path.exists(new_filename):
            return new_filename
        i += 1

#function to append to file
#variables:
#filename = filename
#st = string to write
def f_append_to_file(filename,st):
    """Write a few lines to the file."""
    with open(filename, 'a') as file:
        file.write(st)
        file.write("\n")

# Custom exception for timeout
class TimeoutException(Exception):
    pass

# Signal handler to raise timeout exception
def handler(signum, frame):
    raise TimeoutException("Function execution took too long!")

#monitor functions for timeout exceptions
def monitor_function(func, args, timeout=5):
    # Set the signal handler for timeout
    #catches time exceptions
    signal.signal(signal.SIGALRM, handler)
    # Schedule the alarm to go off after 'timeout' seconds
    signal.alarm(timeout)
    try:
        result = func(args)
    except TimeoutException as e:
        raise
    else:
        signal.alarm(0)
        return result
    finally:     
        signal.alarm(0)


def array_to_filename(arr, prefix="file", suffix=".txt"):
    """
    Transforms an array of floats into a human-readable filename.

    Parameters:
    - arr: The array of floats.
    - prefix: Optional prefix for the filename.
    - suffix: Optional suffix/extension for the filename.

    Returns:
    - A filename-safe string.
    """
    # Round floats for brevity and convert to strings
    float_strings = [f"{x:.2f}" for x in arr]

    # Join the strings with an underscore
    filename_body = "_".join(float_strings)

    # Replace invalid filename characters (e.g., '.' in floats) with safe characters
    filename_body = filename_body.replace(".", "p")

    # Combine with prefix and suffix
    filename = f"{prefix}_{filename_body}{suffix}"
    return filename

#dont go beyond
r_targ=3e-9

def stop_condition(t, X,*args):
    r_max=f_r_max(X,vec_l,vec_c)
    return r_max - r_targ # Stop when y[0] (position) reaches 0

stop_condition.terminal = True  # Tells the solver to terminate on this event

### end of function definitions    
### Start of simulation
### setting parameters

#hardcode some starting params in order to fit data
#new function For U degradation.
#200, 330, 720 to 1,000 μV/h
#average voltage drops recorded from 
#now transformed to secconds!
rates=np.array([200*1e-6,200*1e-6,330*1e-6,720*1e-6,1000*1e-6])/(60*60)
t_rates=np.array([200,200+120,200+120*2,200+120*3,200+120*4])*60*60
#transformed to secconds
U0=0.7
#max tiem = trates[4]
@jit(nopython=True,parallel=isparalel)
def funcUPt_exp(t):
    if t <t_rates[0]:
        U=U0 - rates[0]*t
    elif t<t_rates[1]:
        U=U0 - rates[0]*t_rates[0] - rates[1]*(t-t_rates[0])
    elif t<t_rates[2]:
        U=U0 - rates[0]*t_rates[0] - rates[1]*(t_rates[1]-t_rates[0]) - rates[2]*(t-t_rates[1])       
    elif t<t_rates[3]:
        U=U0 - rates[0]*t_rates[0] - rates[1]*(t_rates[1]-t_rates[0]) - rates[2]*(t_rates[2]-t_rates[1]) - rates[3]*(t-t_rates[2])
    elif t<t_rates[4]:
        U=U0 - rates[0]*t_rates[0] - rates[1]*(t_rates[1]-t_rates[0]) - rates[2]*(t_rates[2]-t_rates[1]) - rates[3]*(t_rates[3]-t_rates[2])- rates[4]*(t-t_rates[3])
    else:
        U=U0 - rates[0]*t_rates[0] - rates[1]*(t_rates[1]-t_rates[0]) - rates[2]*(t_rates[2]-t_rates[1]) - rates[3]*(t_rates[3]-t_rates[2]) - rates[4]*(t_rates[4]-t_rates[3])
    return U

FCstate = [funccHp,funcT,funcUPt_exp]

print("imported all the functions")
print("define parameters")

#actual program from here
#FCstate = [funccHp,funcT,funcUPt_const]
#starting dist  params pre fited
s_c=0.7
loc_c=0.4
scale_c=2

vek=[7.500e-01,1.200e-01,1.978e+00]
#make starting distribution
dist_bins_om ,vec_c, vec_l = f_start_dist_alter(lmin=0.25,lmax=5,cmin=0.25,cmax=14,MPars=ModPars, loc_l=0.25*3, scale_l=0.25, s_c=vek[0], loc_c=vek[1], scale_c=vek[2])

C,L = np.meshgrid(vec_c,vec_l)
### simulation voltage and time parameters
### curently fixed for input CEA voltage profile
"""
rateMax=50*10**-3 #mV/s
Umin=0.6 #V bot
Umax=1.0  #V top
Am=Umax-Umin #amplitude
t0=Am/(rateMax)
t0=1200 #cycle duration in  [s]
#t0=data[-1,-1]
p=2*t0
nCyc = 1 #Number of cycles
tMax=nCyc*t0 #recalculate the final time
#tMax in hours
"""

#change simulation duration
#Nhours=700
#tMax=60*60*Nhours

Nsecs=t_rates[-1]
tMax=Nsecs
iDim = 200-1  #number of observation points at which we save data
dt = tMax/(iDim)

ti = np.arange(0,tMax+dt,dt)

nSt = 3
#number of bins
nc = len(vec_c)
nl = len(vec_l)
# total number of variables
nSys = nSt+nc*nl

### set initial distribution
#initial states
X0 = np.zeros(nSys)
# cathode
#fill the bins starting vector
X0[0]=1.0*10**-10
X0[1]=1.0*10**-2
X0[2]=1.0*10**-10
X0[3:]=dist_bins_om.flatten()

#print("X0")
#print(X0)
print("vec_l")
print(vec_l)

print("vec_c")
print(vec_c)

#vec_c=degradation_model_PtCo_cycle_skip_20240205.vec_c
#vec_l=degradation_model_PtCo_cycle_skip_20240205.vec_l

len_l=len(vec_l)
len_c=len(vec_c)

dl=vec_l[1]-vec_l[0]
dc=vec_c[1]-vec_c[0]

#create X by running sim
tol_r = 1e-4 #tolerances, recommend 1e-5 or 1e-6
tol_a = 1e-4
nm=1e-9

##load inoput data fro fitting: r distribution
mypath=os.getcwd()
print("starting input data fit")
print( "working from dir: " + str(pth.Path(mypath)))

#data=np.loadtxt("UcatData_RealCycCEA.txt",skiprows=2)
#this was CHANGED
#data=np.loadtxt(str(pth.Path(mypath))+"/dataPSD_CEA.txt",skiprows=1)

file_path = "g3c_input.xlsx"  # Update with your actual file path
df = pd.read_excel(file_path, engine="openpyxl",skiprows=1)  # Specify engine for old .xls format
data = df.to_numpy()


#vars=[14,16]
#vek_in=[0.9,3]
ModPars0=np.copy(ModPars)  
#for i,j in enumerate(vars):
#    ModPars0[j]=vek_in[i]

pars=(ModPars0,vec_c,vec_l,FCstate[1],FCstate[2],FCstate[0])

#run a simulation whole time 
print("test one simulation run")

for i, par in enumerate(ModPars0):
    print(str(i)+"\t"+str(par))



"""

t0=time.time()

#sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), X0, method='LSODA',events=None, t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)
sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), X0, method='LSODA',events=stop_condition, t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)

t1=time.time()
print(str(t1-t0))

print(sol0)
X=sol0.y[:,-1]

r_max0=f_r_max(X0,vec_l,vec_c)
print(r_max0)
r_max=f_r_max(X,vec_l,vec_c)
print(r_max)

om=f_trans_x_to_om(X0,3,vec_c,vec_l)
om1=f_trans_x_to_om(X,3,vec_c,vec_l)

stats_X=f_stats(X,ModPars,vec_c,vec_l)
stats_X0=f_stats(X0,ModPars,vec_c,vec_l)

res,vec_r=f_cl_to_rl_rc(X,nSt,vec_l,vec_c)
l,r=f_flat_res_om(res,vec_l,vec_r)  #sum to r distribution
#norm end r dist
rez_norm_r = f_norm_dist1d(vec_r,r)

res0,vec_r=f_cl_to_rl_rc(X0,nSt,vec_l,vec_c)
l0,r0=f_flat_res_om(res0,vec_l,vec_r)  #sum to r distribution
#norm start r dist
rez_norm_r0 = f_norm_dist1d(vec_r,r0)

print("ratio of particles remaining")
print(np.sum(om1*dl*dc)/np.sum(om*dl*dc))

print("surface change")
print("avg r change:")
print([i+(8-len(i))*" " for i in ["num_tot","r_avg","c_avg","l_avg","SA","ESA","c_tot","l_tot","Pt","Cu"]])
print(["{:.2e}".format(i) for i in stats_X0])
print(["{:.2e}".format(i) for i in stats_X])
#print(stats_X0)
#print(stats_X)
print("ratios")
print(["{:.6f}".format(i) for i in np.array(stats_X)/np.array(stats_X0)])

"""


#so far so good
#data inports for fiting function HARDCODED

#prepare data for interpolation
#load data
#data=np.loadtxt(str(pth.Path(mypath))+"/dataPSD_CEA.txt",skiprows=1)
#interpolate to r values
#d1_i=np.interp(vec_r,data[:,0]*nm,data[:,1])
#d2_i=np.interp(vec_r,data[:,0]*nm,data[:,2])
#d3_i=np.interp(vec_r,data[:,0]*nm,data[:,3])

#norm dist values
#d1_n = f_norm_dist1d(vec_r,d1_i)
#d2_n = f_norm_dist1d(vec_r,d2_i)
#d3_n = f_norm_dist1d(vec_r,d3_i)

#add new functions related to dif evo below:

#set callback function for dif evo in order to log results.
iteracija = []
rezultat_konvergence = []
object_vals = []

#create a log file if it doesnt exist 
current_date = datetime.now().strftime('%Y-%m-%d')
base_filename = f"dif_evo_{current_date}.txt"
# Create a unique filename if needed
filename = f_create_unique_filename(base_filename)

#set vars
vars=[0,1,3,7,8,9,10,11]
#vars=[16]
#vars=[14,16,22]
#vars=[14,16]
initial_vars=[]
bounds = []
vars_0=[]

#construct boundaries to check
#if variable is around unity var by half if not var by order
for i in vars:
    #make first guess
    vars_0.append(ModPars[i])
    #construct bounds
    if 0.5 < ModPars[i] <=1.5:
        bounds.append((ModPars[i]*0.8,ModPars[i]*1.2))
        initial_vars.append(ModPars[i])
    elif 1.5 < ModPars[i] <3:
        bounds.append((ModPars[i]*0.5,ModPars[i]*2))
        initial_vars.append(ModPars[i])
    elif 0.01 < ModPars[i] <0.5:
        bounds.append((ModPars[i]*0.5,ModPars[i]*2))
        initial_vars.append(ModPars[i])
    else:
        bounds.append((ModPars[i]*0.5,ModPars[i]*2))
        initial_vars.append(ModPars[i])

#bounds=[(0.8, 1.2), (1, 3), (0.01e9, 0.04e9)]
#bounds=[(0.8, 1.2), (1, 3)]
#bounds=[(1, 5)]

#calback function in dif evo
#defined in main for access to global
def f_callback(xk, convergence):
    #set when simulation is run
    global iteracija
    global rezultat_konvergence
    global object_vals
    #at each stage append iteration num, variables and objective function vals.
    iteracija.append(len(iteracija)+1)
    rezultat_konvergence.append(xk)
    object_vals.append(convergence)
    f_append_to_file(filename,str(xk)+" "+str(convergence) )
    print(f"Current solution: {xk}, Convergence: {convergence}")

#global evaluacija
#evaluacija=0

#must define price in inport main since it needs to know vec and can take it as a parameter
def f_price(vek_in):

    t0=time.time()
    #important make a shaow copy
    ModPars0=np.copy(ModPars)  
    for i,j in enumerate(vars):
        ModPars0[j]=vek_in[i]

    pars=(ModPars0,vec_c,vec_l,FCstate[1],FCstate[2],FCstate[0])
    #solve a run
    #sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), X0, method='LSODA',events=stop_condition, t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)
    sol0 = solve_ivp(f_dfdt, (ti[0],ti[-1]), X0, method='LSODA',events=None, t_eval=ti, args=pars, rtol  = tol_r, atol = tol_a)

    #get X from the solution
    X=sol0.y[:,-1]
    
    #transform to r distribution
    #res,vec_r=f_cl_to_rl_rc(X,nSt,vec_l,vec_c)
    om=f_X_to_dist(X,vec_l,vec_c)
    r=f_om_to_r(om)
    #l,r=f_flat_res_om(res,vec_l,vec_r)  #sum to r distribution
    #dl2 = (vec_l[1] - vec_l[0]) * 0.5 
    dc2 = (vec_c[1] - vec_c[0]) * 0.5
    #does it have to be that way? would it work with any dr?
    dr2 = dc2
    #dr vector
    vec_r=np.arange(dr2,vec_l[-1]+vec_c[-1],2*dr2)
    #norm end r dist
    rez_norm_r = f_norm_dist1d(vec_r,r)
    
    #calculate the price function
    #d2_n calculated before, its a hardcoded input for all evolution calls
    price = f_sum_abs(d2_n,rez_norm_r,vec_r)
    #optionaly print price and input vector values
    #print(price)
    #print(vek_in)
    #print(price)
    #evaluacija=evaluacija+1
    #print("in m_f_price "+str(rez)+"\n"+str(vek_in) +"\n"+str(evaluacija))

    print("in f_price "+str(vek_in) + "\n"+str(price))
    
    fname=str(pth.Path(mypath))+"/"+array_to_filename(vek_in,prefix="slika", suffix=".png")
    print(fname)
    
    t1=time.time()
    print(str(t1-t0))
    
    #plt.plot(vec_r,r)
    #plt.savefig( fname , dpi=300, bbox_inches='tight')
    #plt.close()  

    return price

#monitor price function 
def m_f_price(vek_in):
    try:
        #input simlatin funcion
        result = monitor_function(f_price, args=vek_in, timeout=50)
    except TimeoutException as e:
        #return high price in timeout
        print(f"zmanjkal casa: {e}")
        rez=10
    except Exception as e:
        #return high price in error
        print("other error")
        print(e)
        rez=20
    else:
        #rez[i]=(result-r0)/r0
        rez=result
    finally:
        # Optional: You can add cleanup code here if needed
        pass

    #print("in m_f_price "+str(vek_in) + "\n"+str(rez))
    return rez

def imported_main():

    print("This is the imported module's main function.")
    print("dif evo fiting")
    #print vars
    print("vars")
    print(vars)
 
    #print bounds
    print("bounds")
    print(bounds)   
    """
    #check extreme bounds
    print("check price function,low range")
    low_values = [t[0] for t in bounds]
    price=f_price(low_values)
    #print(price)

    print("check price function,high range")
    #vek_in=[1.]
    high_values = [t[-1] for t in bounds]
    price=f_price(high_values)
    #print(price)
    """

    #print to files
    f_append_to_file(filename,str(ModPars))
    f_append_to_file(filename,str(vars))
    f_append_to_file(filename,str(bounds))

    print("starting diff evo")
    #make diff evo
    #this works
    #rezultat = differential_evolution(f_price, bounds,x0=None,events=None,popsize=20, callback=None, tol=0.0001, disp = True,updating="deferred",workers=16)
    #rezultat = differential_evolution(f_price, bounds,x0=[9.11963145e-01, 1.90612536e+00, 1.98431860e+07],popsize=20, callback=None, tol=0.0001, disp = True,updating="deferred",workers=16)
    rezultat = differential_evolution(f_price, bounds,x0=vars_0,popsize=1000, callback=f_callback, tol=0.0001, disp = True,updating="deferred",workers=16)

    #rezultat = differential_evolution(m_f_price, bounds,x0=initial_vars, callback=f_callback, tol=0.0001, disp = True,updating="deferred",workers=10)

# Guard to prevent automatic execution when imported
if __name__ == "__main__":
    print("The script is running directly!")
    imported_main()
else:
    print("The script is being imported, and nothing will execute.")

