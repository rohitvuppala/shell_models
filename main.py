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
            g = np.conjugate(c1*u[i+1]*u[i+2] + c2*u[n-1]*u[i+1] + c3*u[n-1]*u[n-2])*1j
        elif i == 1:
            g = np.conjugate(c1*u[i+1]*u[i+2] + c2*u[i-1]*u[i+1] + c3*u[i-1]*u[n-1])*1j
        elif i == n-2:
            g = np.conjugate(c1*u[i+1]*u[0] + c2*u[i-1]*u[i+1] + c3*u[i-1]*u[i-2])*1j
        elif i == n-1:
            g = np.conjugate(c1*u[0]*u[1] + c2*u[i-1]*u[0] + c3*u[i-1]*u[i-2])*1j
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
u0[4] = 1e-5*(1 + 1j)
u0[6] = 1e-5*(1 + 1j) 
#init time
t0    = 0.0

# ODE solver parameters
#abserr = 1.0e-8
#relerr = 1.0e-6
tend      = 100.0
numpoints = 1001
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
                print('{:.2f}'.format(i),end=" ",file=out_file)
            print('\n',file=out_file)
            npaa.append(np.append(r.t,r.y))
    #f.write("\n")
    
#f.close()   

#%%r
data = np.load("data.npz", mmap_mode="r")

print(data)



