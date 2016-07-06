# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:19:00 2015

@author: Ali Abbasi [aliabbasi.civileng@gmail.com]
This program calculate & plot the inflow boundart profiles considering the atmospheric stability conditions
"""

import pandas as pd  
import matplotlib.pyplot as plt
import pylab 
import scipy # Scientific Python functions
import pylab # Plotting functions
import math
import numpy as np
import matplotlib.dates as mdates
from matplotlib import rc, rcParams
from datetime import datetime, timedelta
import pdb
from scipy import stats
#-----------------------------------------------------------
def inlet_profile(stability,axis,z):
    rho_a=1.186
    cp_a=1003.62
    g=9.81
    z0=0.002
    K=0.41
    
    L_neutral=10000000
    L_stable=309.50
    L_unstable=-108.10
    
    T0_neutral=298.15
    T0_stable=283.15
    T0_unstable=313.15
    
    q_neutral=0.00
    q_stable=-30.00
    q_unstable=100.00
    
    uStar_neutral=0.481
    uStar_stable=0.472
    uStar_unstable=0.497
    if (stability=="neutral"):
        L= L_neutral
        T0=T0_neutral
        q= q_neutral
        uStar= uStar_neutral
    #z=10#495;#0.0171 #500
    if (stability=="stable"):
        L= L_stable
        T0=T0_stable
        q= q_stable
        uStar= uStar_stable
    if (stability=="unstable"):
        L= L_unstable
        T0=T0_unstable
        q= q_unstable
        uStar= uStar_unstable
        
    #z = np.linspace(0.0, 500.0, num=10000)
    #x = np.linspace(0.0, 5000.0, num=10000)
    
    if (L<0):
        x=np.power((1-16*z/L),0.25)
        phi_m_n=np.power((1-16*z/L),-0.25)
        phi_h_n=2*np.log((1+np.power(x,2))/2)
        phiU_m_n=np.log(((1+np.power(x,2))/2)*np.power((1+x)/2,2))-2*np.arctan(x)+np.pi/2
    phi_m_p=1+5*z/L
    if (L>0):
        phi_m = phi_m_p
    else:
        phi_m = phi_m_n
    phi_e_n=1-z/L
    phi_e_p=phi_m-z/L
    if (L>0):
        phi_e = phi_e_p
    else:
         phi_e = phi_e_n
    
    #phi_h_n=2*np.log((1+np.power(x,2))/2)
    #print phi_h_n
    phi_h_p=-5*z/L
    if (L>0):
        phi_h=phi_h_p
    else:
        phi_h= phi_h_n
    
    #phiU_m_n=np.log(((1+np.power(x,2))/2)*np.power((1+x)/2,2))-2*np.arctan(x)+np.pi/2
    phiU_m_p =-5*z/L
    if (L>0):
        phiU_m= phiU_m_p
    else:
        phiU_m= phiU_m_n
    
    TStar=-q/(rho_a*cp_a*uStar)
    T_inlet=(TStar/K)*(np.log(z/z0)-phi_h)+T0 - g/cp_a*z
    epsilon_inlet=np.power(uStar,3)/(z*K)*phi_e
    k_inlet=5.48*np.power(uStar,2)*np.sqrt(phi_e/phi_m)
    u_inlet=(uStar/K)*(np.log(z/z0)-phiU_m)
    
    if (axis=="z"):
        y = np.around(z,decimals=5)
        inlet_profile = pd.DataFrame(y,columns=['z'], index=z)
        inlet_profile['T_inlet']= T_inlet
        inlet_profile['epsilon_inlet']= epsilon_inlet
        inlet_profile['k_inlet']= k_inlet
        inlet_profile['u_inlet']= u_inlet
    else:
        x = np.linspace(0.0, 5000.0, num=10000)
        y = np.around(x,decimals=5)
        inlet_profile = pd.DataFrame(y,columns=['x'], index=x)
        inlet_profile['T_inlet']= T_inlet
        inlet_profile['epsilon_inlet']= epsilon_inlet
        inlet_profile['k_inlet']= k_inlet
        inlet_profile['u_inlet']= u_inlet
    
    #print inlet_profile.describe()
    #plt.plot(inlet_profile.T_inlet, inlet_profile.index)
    return inlet_profile

#print "epsilon  %10.8f" % epsilon_inlet
#print "k  %10.8f" % k_inlet
#print "T  %10.6f" % T_inlet
#print "U  %10.6f" % u_inlet
#nut = 0.0333*np.power(k_inlet,2) / epsilon_inlet
#print "nut  %10.5f" % nut
#z = np.linspace(-10.0, 10.0, num=100)
#df1 = inlet_profile("unstable","z",z)
#print df1
#x = np.linspace(0.0, 5000.0, num=10000)
#y= np.around(x,decimals=5)
#print y[11]

#plt.plot(df1['k_inlet'], df1.z)
#plt.plot(z, np.tanh(z))



rho_w=1000
cp_w=4180
eta_measuerd = 3.0
def Q_swr_z(z, eta_m):
    f= scipy.zeros(7)
    eta =scipy.zeros(7)
    f_times_eta =scipy.zeros(7)
    f_times_eta_div =scipy.zeros(7)
    f[0]=0.046
    eta[0]=eta_m
    f[1]=0.43
    eta[1]=eta_m
    f[2]= 0.214
    eta[2]=2.92
    f[3]=0.02
    eta[3]=20.4
    f[4]=0.089
    eta[4]=29.5
    f[5]=0.092
    eta[5]=98.4
    f[6]=0.109
    eta[6]=2880
    for i in range(0,np.size(eta)):
        f_times_eta[i] = f[i] * np.exp(-eta[i]*z)
        f_times_eta_div[i] = -eta[i] * f_times_eta[i]
        #print f_times_eta_div[i]
    Q = sum(f_times_eta_div)    
    return Q

Rs = 800
Q_swr_0 = 800
z = np.linspace(0.0, 0.25, num=1001)
Q_swr = scipy.zeros(np.size(z))
Q_swr_1 = scipy.zeros(np.size(z))
Q_swr_2 = scipy.zeros(np.size(z))
eta_measuerd = 1.0

for i in range(0,np.size(z)):
    Q_swr_1[i]=-1* Q_swr_0 *Q_swr_z(z[i],eta_measuerd)/(cp_w*rho_w)
    print z[i], Q_swr_1[i]
plt.plot(Q_swr_1[0:100],z[0:100])

eta_measuerd = 10.0

for i in range(0,np.size(z)):
    Q_swr_2[i]=-1* Q_swr_0 *Q_swr_z(z[i],eta_measuerd)/(cp_w*rho_w)
    print z[i], Q_swr_2[i] 
plt.plot(Q_swr_2[0:100],z[0:100])







