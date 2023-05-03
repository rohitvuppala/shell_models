#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:43:59 2023

@author: Rohit K S S Vuppala
         Graduate Student, 
         Mechanical and Aerospace Engineering,
         Oklahoma State University.

@email: rvuppal@okstate.edu

"""

import matplotlib.pyplot as plt
import numpy as np
# Use COMPLEX_ODE to solve the differential equations defined by the vector field
from scipy.integrate import complex_ode,odeint
from npy_append_array import NpyAppendArray


plt.style.use('seaborn-poster')

#
# Scipy has a bug while solving complex odes with arguments - any other way might not be reliable 
#
"""
Create a class to pass the function and jacobian function if available

Arguments:
    f     : function which has the rhs of the system arranged with lhs as 
        first order derivatives for the variables
   jac    : function for jacobian if known
   fargs  : any values that need to be passed to the function in a list
   jacargs: any values that need to be passed to the jacobian function in a list    
"""
class goy(object):
    def __init__(self, f, fargs=[]):#jac, fargs=[],jacargs=[]):
        
        self._f     = f
        #self._jac   = jac
        self.fargs  = fargs
        #self.jacargs= jacargs
        
        
    def f(self, t, y):
        return self._f(t,y, *self.fargs)
    
    # def jac(self, t, y):
    #     return self._jac(t,y, *self.jacargs)
"""
Defines the differential equations for the GOY shell model.

Arguments:
    u :  vector of the state variables:
              u = [u0,u1,u2,....,u21]
    t :  time
    p :  vector of the parameters:
              p = [lambda,k0,eps,nu,f,alpha]
"""    
def vectorfield(t, u, *p):

    n = len(u)
    
    lmbda, k0, eps, nu, f0, alpha = p

    # Create rhs for lhs = (u0',u1',u2',u3',...,u21'):
    rhs = []
    #
    # Define a function for computing k_i
    #
    def k(i):
        return k0*(lmbda**(i+1))

    for i in range(n):
        
        
        #########################################################################################
        # Note : the values are stored from index starting 0, however shell number starts at 1
        #########################################################################################
        # Ghost values for u[-1],u[-2],u[22],u[23]
        
        #shell number
        sn = i + 1; 
        
        #compute dissipative term
        d = (-alpha*nu*(k(sn)**2)*u[i])
        
        #compute source term
        if sn == 4:
            f = f0            # generally, 5e-3*(1+1j)
        else:
            f = 0
        
        #compute the non-linear advection term
        
        if sn == 1:
            c1 = k(sn)
            c2 = 0
            c3 = 0
        elif sn == 2:
            c1 = k(sn)
            c2 =-(eps*k(sn-1))
            c3 = 0
        elif sn == n-1:
            c1 = 0
            c2 =-(eps*k(sn-1))
            c3 = (eps-1)*k(sn-2)
        elif sn == n:
            c1 = 0
            c2 = 0
            c3 = (eps-1)*k(sn-2)
        else:
            c1= k(sn)
            c2=-(eps*k(sn-1))
            c3= (eps-1)*k(sn-2)
        
        if i == 0:
            g = np.conjugate(c1*u[i+1]*u[i+2])*1j
        elif i == 1:
            g = np.conjugate(c1*u[i+1]*u[i+2] + c2*u[i-1]*u[i+1])*1j
        elif i == n-2:
            g = np.conjugate(c2*u[i-1]*u[i+1] + c3*u[i-1]*u[i-2])*1j
        elif i == n-1:
            g = np.conjugate(c3*u[i-1]*u[i-2])*1j
        else:
            g = np.conjugate(c1*u[i+1]*u[i+2] + c2*u[i-1]*u[i+1] + c3*u[i-1]*u[i-2])*1j
        
        
        
        rhs.append(d+f+g)
    
    return rhs
"""
Define a jacobian function

Arguments:
    u :  vector of the state variables:
              u = [u0,u1,u2,....,u21]
    t :  time
    p :  vector of the parameters:
              p = [lambda,k0,eps,nu,f,alpha]
"""
def jac(t, u, *p):
    return []


#%%
#
# Solve for the system
#

#Number of shell models
n = 22

# Parameter values
lmbda = 2       + 0j
k0    = 2**(-4) + 0j
eps   = 0.5     + 0j
nu    = 1e-8    + 0j
f0    = 5e-3*(1 + 1j)    
alpha = 1       + 0j

#
# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
#
u0 = []
for i in range(n):
    u0.append(0.0)
#3rd and 5th shells perturbed
#########################################################################################
# Note : the values are stored from index starting 0, however shell number starts at 1
#########################################################################################
u0[2] = 1e-5*(1 + 1j)
u0[4] = 1e-5*(1 + 1j) 
#init time
t0    = 0.0

# ODE solver parameters
#abserr = 1.0e-8
#relerr = 1.0e-6
tend      = 1200.0
numpoints = 3001
dt        = (tend-t0) / (numpoints - 1)

# Pack up the parameters:
p = [ lmbda, k0, eps, nu, f0, alpha ]

#
# Call the Complex_ODE solver using the class we created
#
#case = goy(vectorfield, jac, fargs=p[:], jacargs=[])
case = goy(vectorfield, fargs=p)
r = complex_ode(case.f)#, case.jac)
r.set_initial_value(u0,t0)
r.set_integrator('dop853')
#
# Integration
#
#### FIX THIS WRITING TO A FILE LINE BY LINE#####
f = open(r"data.dat","w")
f.close()
with open(r"data.dat","a") as out_file:
    with NpyAppendArray("data.npz", delete_if_exists=True) as npaa:
        while r.successful() and r.t < tend:
            r.integrate(r.t+dt)
            print('{:.2f}'.format(r.t),end=" ",file=out_file)
            print('{:.2f}'.format(r.t))
            for i in r.y:
                print('{:.2E}'.format(i),end=" ",file=out_file)
            print('\n',end="",file=out_file)
            npaa.append(np.append(r.t,r.y))
    #f.write("\n")
    
#f.close()   

#%%r
#
#Everything written the file as continuous array so need to unwrap them as needed
#

data   = np.load("data.npz", mmap_mode="r")

tarray = data[0::n+1]
nview  = numpoints-1
darray = np.zeros((nview,n),dtype=np.complex128)
for i in range(n):
    darray[:,i] = data[i+1:nview*(n+1):n+1]
    
for i in range(n):
    plt.xlabel("Time")
    plt.ylabel("Real part")
    plt.plot(tarray[:nview],np.real(darray[:nview,i]),label="mode:"+str(i+1))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=6, fancybox=True, shadow=True)
plt.savefig("real.png",dpi=300,bbox_inches='tight')

plt.figure()
for i in range(n):
    plt.xlabel("Time")
    plt.ylabel("Img part")
    plt.plot(tarray[:nview],np.imag(darray[:nview,i]),label="mode:"+str(i+1))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=6, fancybox=True, shadow=True)
plt.savefig("imag.png",dpi=300,bbox_inches='tight')

#%%
#
# Compute Energy E_i = 1/2 |U_i|^2/k_i^2 
#

#Take the solution from the last timestep 
E      = np.zeros(n,dtype=np.double)
k      = np.zeros(n,dtype=np.double)
E_comp = np.zeros(n,dtype=np.double)
for i in range(n):
    E[i]  = 0.5* np.absolute(r.y[i])**2/(k0*(lmbda**(i+1)))**2         #Since shell number starts from 1 but storage index starts from 0
    k[i]  = np.real(k0)*(np.real(lmbda)**(i+1))
    E_comp[i]= ((np.real(k0)*(np.real(lmbda)**(i+1)))**2)**(-5/3)

fig = plt.figure()
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.scatter(k,E,color='green',label='Energy spectrum N=22')
ax.plot(k,E_comp,color='k',linestyle='--', label='Line with slope -5/3')
ax.set_xlabel("k",fontsize=24)
ax.set_ylabel("E(k)",fontsize=24)
ax.legend(fontsize=24)
plt.savefig("Energy-spectrum-plot.png",dpi=300)



