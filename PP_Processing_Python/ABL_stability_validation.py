# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:01:24 2015

@author: Ali abbasi [aliabbasi.civileng@gmail.com]
This programm reads the sample files of different cases and plot the errors to compare!
"""
# Imprting necessary Libraries
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

#------------------------------------------------------------------
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
#----------------------------------------------------------------------
pylab.rc('text', usetex=True)
# Change default settings for figures
newdefaults = {'font.family' : 'serif',
               'font.serif': 'Times New Roman',
               #'fontname':    'Times New Roman',  # Use Arial font
               'backend':       'tif',  # Save figure as EPS file   
               'axes.labelsize':   50,  # Axis label size in points
               'text.fontsize':    50,  # Text size in points
               'legend.fontsize':  50,  # Legend label size in points
               'xtick.labelsize':  50,  # x-tick label size in points
               'ytick.labelsize':  50,  # y-tick label size in points
               'lines.markersize': 50,  # markersize, in points
               'lines.linewidth':   3.5 # line width in points
               }
pylab.rcParams.update(newdefaults)
#-------------------------------------------------------
# Definig the Class to read CSVs
class reading_z_sample:
   'Common base class for all files'
   sampleCount = 0

   def __init__(self, stability,axis_sample,name1,name_U1,name2,name_U2,name3,name_U3,name4,name_U4,name5,name_U5,name6,name_U6):
      self.name1 = name1
      self.name2 = name2
      self.name3 = name3
      self.name4 = name4
      self.name5 = name5
      self.name6 = name6
      self.name_U1 = name_U1
      self.name_U2 = name_U2
      self.name_U3 = name_U3
      self.name_U4 = name_U4
      self.name_U5 = name_U5
      self.name_U6 = name_U6
      self.stability = stability
      self.axis_sample = axis_sample
      reading_z_sample.sampleCount += 1
   
   def displayCount(self):
     print "Total Sample Files %d" % reading_z_sample.sampleCount

   def display_sampleFiles(self):
      print "File of U : ", self.name_U1, "    Files of others :",self.name1 
      
   def reading_CSV(self):
     delimitsign = ',' #'\t' # For tab (\t) delimited file
     commentsign='#'
     # Reading .csv files for differnt parameters
     # Note the path of the file that is read!!
     df_Data_sample_eps_1= pd.read_csv(self.name1,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_1','epsilon_1','k_1','nut_1','p_1'] , header=0)
     df_Data_sample_U_1= pd.read_csv(self.name_U1,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_1','U_1_1','U_2_1'] , header=0)
     df_Data_sample_eps_2= pd.read_csv(self.name2,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_2','epsilon_2','k_2','nut_2','p_2'] , header=0)
     df_Data_sample_U_2= pd.read_csv(self.name_U2,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_2','U_1_2','U_2_2'] , header=0)
     df_Data_sample_eps_3= pd.read_csv(self.name3,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_3','epsilon_3','k_3','nut_3','p_3'] , header=0)
     df_Data_sample_U_3= pd.read_csv(self.name_U3,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_3','U_1_3','U_2_3'] , header=0)
     df_Data_sample_eps_4= pd.read_csv(self.name4,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_4','epsilon_4','k_4','nut_4','p_4'] , header=0)
     df_Data_sample_U_4= pd.read_csv(self.name_U4,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_4','U_1_4','U_2_4'] , header=0)
     df_Data_sample_eps_5= pd.read_csv(self.name5,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_5','epsilon_5','k_5','nut_5','p_5'] , header=0)
     df_Data_sample_U_5= pd.read_csv(self.name_U5,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_5','U_1_5','U_2_5'] , header=0)
     df_Data_sample_eps_6= pd.read_csv(self.name6,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_6','epsilon_6','k_6','nut_6','p_6'] , header=0)
     df_Data_sample_U_6= pd.read_csv(self.name_U6,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_6','U_1_6','U_2_6'] , header=0)
     df_Data_sample = pd.concat([df_Data_sample_eps_1, df_Data_sample_U_1,df_Data_sample_eps_2, df_Data_sample_U_2,df_Data_sample_eps_3, df_Data_sample_U_3 \
                       ,df_Data_sample_eps_4, df_Data_sample_U_4,df_Data_sample_eps_5, df_Data_sample_U_5,df_Data_sample_eps_6, df_Data_sample_U_6], axis=1)     
     #df_Data_sample=df_Data_sample_eps.append(df_Data_sample_U,ignore_index = True) 
     z = np.linspace(0.0, 500.0, num=9999)     
     df_Data_inlet = inlet_profile(self.stability,self.axis_sample,z)
     df_Data_inlet.index=df_Data_sample.index 
     df_Data_sample['eps_err_1'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_1'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['eps_err_2'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_2'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['eps_err_3'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_3'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['eps_err_4'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_4'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['eps_err_5'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_5'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['eps_err_6'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_6'])/(df_Data_inlet['epsilon_inlet']))*100

     df_Data_sample['T_err_1'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_1'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['T_err_2'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_2'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['T_err_3'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_3'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['T_err_4'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_4'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['T_err_5'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_5'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['T_err_6'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_6'])/(df_Data_inlet['T_inlet']))*100

     df_Data_sample['k_err_1'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_1'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['k_err_2'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_2'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['k_err_3'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_3'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['k_err_4'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_4'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['k_err_5'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_5'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['k_err_6'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_6'])/(df_Data_inlet['k_inlet']))*100

     df_Data_sample['u_err_1'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_1'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['u_err_2'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_2'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['u_err_3'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_3'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['u_err_4'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_4'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['u_err_5'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_5'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['u_err_6'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_6'])/(df_Data_inlet['u_inlet']))*100
     
#     df_Data_sample['nut_err_1'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_1'])/(df_Data_sample['nut_1']))*100
#     df_Data_sample['nut_err_2'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_2'])/(df_Data_sample['nut_1']))*100
#     df_Data_sample['nut_err_3'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_3'])/(df_Data_sample['nut_1']))*100
#     df_Data_sample['nut_err_4'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_4'])/(df_Data_sample['nut_1']))*100
#     df_Data_sample['nut_err_5'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_5'])/(df_Data_sample['nut_1']))*100
#     df_Data_sample['nut_err_6'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_6'])/(df_Data_sample['nut_1']))*100
#     
#     df_Data_sample['p_err_1'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_1'])/(df_Data_sample['p_1']))*100
#     df_Data_sample['p_err_2'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_2'])/(df_Data_sample['p_1']))*100
#     df_Data_sample['p_err_3'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_3'])/(df_Data_sample['p_1']))*100
#     df_Data_sample['p_err_4'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_4'])/(df_Data_sample['p_1']))*100
#     df_Data_sample['p_err_5'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_5'])/(df_Data_sample['p_1']))*100
#     df_Data_sample['p_err_6'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_6'])/(df_Data_sample['p_1']))*100
     
     print df_Data_sample.describe()
     return df_Data_sample
     
#   def define_newDataFile(self):
#       small_value=0.00000001
#       df_totalData = pd.DataFrame(index= self.reading_CSV())
#       df_totalData.index = self.reading_CSV().index
#       df_totalData['epsilon'] = self.reading_CSV()['epsilon']
#       df_totalData['Temp'] = self.reading_CSV()['T']
#       df_totalData['k'] = self.reading_CSV()['k']
#       df_totalData['nut'] = self.reading_CSV()['nut']
#       df_totalData['p'] = self.reading_CSV()['p']
#       df_totalData['Ux'] = self.reading_CSV()['U_0']
#       df_totalData['Uy'] = self.reading_CSV()['U_1']
#       df_totalData['Uz'] = self.reading_CSV()['U_2']
#       df_totalData['Temp-error'] = np.fabs((df_totalData.Temp[0] - df_totalData['Temp'])/(df_totalData.Temp[0]))*100
#       df_totalData['p-error'] = np.fabs((df_totalData.p[0] - df_totalData['p'])/(df_totalData.p[0]))*100
#       df_totalData['nut-error'] = np.fabs((df_totalData.nut[0] - df_totalData['nut'])/(df_totalData.nut[0]))*100
#       df_totalData['k-error'] = np.fabs((df_totalData.k[0] - df_totalData['k'])/(df_totalData.k[0]))*100
#       df_totalData['epsilon-error'] = np.fabs((df_totalData.epsilon[0] - df_totalData['epsilon'])/(df_totalData.epsilon[0]))*100
#       df_totalData['Ux-error'] = np.fabs((df_totalData.Ux[0] - df_totalData['Ux'])/(df_totalData.Ux[0]))*100
#       #df_totalData['Uy-error'] = np.fabs((df_totalData.Uy[0] - df_totalData['Uy'])/(df_totalData.Uy[0]+small_value))*100
#       df_totalData['Uz-error'] = np.fabs((df_totalData.Uz[0] - df_totalData['Uz'])/(df_totalData.Uz[0]+small_value))*100
#       #print df_totalData.describe()
#       return df_totalData
#-------------------------------------------------------
# Definig the Class to read CSVs on X axis
class reading_x_sample:
   'Common base class for all files'
   sampleCount = 0

   def __init__(self, stability, axis_sample,name1,name_U1,name2,name_U2,name3,name_U3):
      self.name1 = name1
      self.name2 = name2
      self.name3 = name3
      self.name_U1 = name_U1
      self.name_U2 = name_U2
      self.name_U3 = name_U3
      self.stability = stability
      self.axis_sample = axis_sample
      reading_x_sample.sampleCount += 1
   
   def displayCount(self):
     print "Total Sample Files %d" % reading_x_sample.sampleCount

#   def error_withInlet(self):
#       self.error = np.fabs((df_totalData.Temp[0] - df_totalData['Temp'])/(df_totalData.Temp[0]))*100
#      print "File of U : ", self.name_U1, "    Files of others :",self.name1 
#      
   def reading_CSV(self):
     delimitsign = ',' #'\t' # For tab (\t) delimited file
     commentsign='#'
     # Reading .csv files for differnt parameters
     # Note the path of the file that is read!!
     df_Data_sample_eps_1= pd.read_csv(self.name1,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_1','epsilon_1','k_1','nut_1','p_1'] , header=0)
     df_Data_sample_U_1= pd.read_csv(self.name_U1,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_1','U_1_1','U_2_1'] , header=0)
     df_Data_sample_eps_2= pd.read_csv(self.name2,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_2','epsilon_2','k_2','nut_2','p_2'] , header=0)
     df_Data_sample_U_2= pd.read_csv(self.name_U2,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_2','U_1_2','U_2_2'] , header=0)
     df_Data_sample_eps_3= pd.read_csv(self.name3,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T_3','epsilon_3','k_3','nut_3','p_3'] , header=0)
     df_Data_sample_U_3= pd.read_csv(self.name_U3,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0_3','U_1_3','U_2_3'] , header=0)
     df_Data_sample = pd.concat([df_Data_sample_eps_1, df_Data_sample_U_1,df_Data_sample_eps_2, df_Data_sample_U_2,df_Data_sample_eps_3, df_Data_sample_U_3], axis=1)     
     #df_Data_sample=df_Data_sample_eps.append(df_Data_sample_U,ignore_index = True) 
     #print df_Data_sample.describe()
     z=100     
     df_Data_inlet = inlet_profile(self.stability,self.axis_sample,z)
     df_Data_inlet.index=df_Data_sample.index     
     #print  df_Data_inlet['epsilon_inlet'] 
     #df_Data_sample_1.index=df_Data_sample.index
     #df_Data_sample['eps_err_1'] = df_Data_inlet.epsilon_inlet 
     #print  df_Data_sample['eps_err_1'] 
     #df_Data_sample['eps_err_1'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_1'])//(df_Data_inlet['epsilon_inlet']))*100
     #df_Data_sample['eps_err_1'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_1'])//(df_Data_inlet['epsilon_inlet']))*100
     
     df_Data_sample['eps_err_1'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_1'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['T_err_1'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_1'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['k_err_1'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_1'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['u_err_1'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_1'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['nut_err_1'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_1'])/(df_Data_sample['nut_1']))*100
     df_Data_sample['p_err_1'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_1'])/(df_Data_sample['p_1']))*100
     #print df_Data_inlet
     #print df_Data_sample['k_err_1']
     z=10     
     df_Data_inlet = inlet_profile(self.stability,self.axis_sample,z)
     df_Data_inlet.index=df_Data_sample.index 
     df_Data_sample['eps_err_2'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_2'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['T_err_2'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_2'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['k_err_2'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_2'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['u_err_2'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_2'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['nut_err_2'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_2'])/(df_Data_sample['nut_1']))*100
     df_Data_sample['p_err_2'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_2'])/(df_Data_sample['p_1']))*100

     z=2     
     df_Data_inlet = inlet_profile(self.stability,self.axis_sample,z)
     df_Data_inlet.index=df_Data_sample.index 
     df_Data_sample['eps_err_3'] = np.fabs((df_Data_inlet['epsilon_inlet'] - df_Data_sample['epsilon_3'])/(df_Data_inlet['epsilon_inlet']))*100
     df_Data_sample['T_err_3'] = np.fabs((df_Data_inlet['T_inlet'] - df_Data_sample['T_3'])/(df_Data_inlet['T_inlet']))*100
     df_Data_sample['k_err_3'] = np.fabs((df_Data_inlet['k_inlet'] - df_Data_sample['k_3'])/(df_Data_inlet['k_inlet']))*100
     df_Data_sample['u_err_3'] = np.fabs((df_Data_inlet['u_inlet'] - df_Data_sample['U_0_3'])/(df_Data_inlet['u_inlet']))*100
     df_Data_sample['nut_err_3'] = np.fabs((df_Data_sample['nut_1'] - df_Data_sample['nut_3'])/(df_Data_sample['nut_1']))*100
     df_Data_sample['p_err_3'] = np.fabs((df_Data_sample['p_1'] - df_Data_sample['p_3'])/(df_Data_sample['p_1']))*100     


     return df_Data_sample
#---------------------------------------------------------------
# Definig the Class to read CSVs
class reading_sample:
   'Common base class for all files'
   sampleCount = 0

   def __init__(self, name,name_U):
      self.name = name
      self.name_U = name_U
      reading_sample.sampleCount += 1
   
   def displayCount(self):
     print "Total Sample Files %d" % reading_sample.sampleCount

   def display_sampleFiles(self):
      print "File of U : ", self.name_U, "    Files of others :",self.name 
      
   def reading_CSV(self):
     delimitsign = ',' #'\t' # For tab (\t) delimited file
     commentsign='#'
     # Reading .csv files for differnt parameters
     # Note the path of the file that is read!!
     df_Data_sample_eps= pd.read_csv(self.name,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','T','epsilon','k','nut','p'] , header=0)
     df_Data_sample_U= pd.read_csv(self.name_U,na_values=['-9999.0000'], index_col=0, sep=",",skiprows=0, names =['x','U_0','U_1','U_2'] , header=0)
     df_Data_sample = pd.concat([df_Data_sample_eps, df_Data_sample_U], axis=1)     
     #df_Data_sample=df_Data_sample_eps.append(df_Data_sample_U,ignore_index = True) 
     #print df_Data_sample.describe()
     return df_Data_sample
     
   def define_newDataFile(self):
       small_value=0.00000001
       df_totalData = pd.DataFrame(index= self.reading_CSV())
       df_totalData.index = self.reading_CSV().index
       df_totalData['epsilon'] = self.reading_CSV()['epsilon']
       df_totalData['Temp'] = self.reading_CSV()['T']
       df_totalData['k'] = self.reading_CSV()['k']
       df_totalData['nut'] = self.reading_CSV()['nut']
       df_totalData['p'] = self.reading_CSV()['p']
       df_totalData['Ux'] = self.reading_CSV()['U_0']
       df_totalData['Uy'] = self.reading_CSV()['U_1']
       df_totalData['Uz'] = self.reading_CSV()['U_2']
       df_totalData['Temp-error'] = np.fabs((df_totalData.Temp[0] - df_totalData['Temp'])/(df_totalData.Temp[0]))*100
       df_totalData['p-error'] = np.fabs((df_totalData.p[0] - df_totalData['p'])/(df_totalData.p[0]))*100
       df_totalData['nut-error'] = np.fabs((df_totalData.nut[0] - df_totalData['nut'])/(df_totalData.nut[0]))*100
       df_totalData['k-error'] = np.fabs((df_totalData.k[0] - df_totalData['k'])/(df_totalData.k[0]))*100
       df_totalData['epsilon-error'] = np.fabs((df_totalData.epsilon[0] - df_totalData['epsilon'])/(df_totalData.epsilon[0]))*100
       df_totalData['Ux-error'] = np.fabs((df_totalData.Ux[0] - df_totalData['Ux'])/(df_totalData.Ux[0]))*100
       #df_totalData['Uy-error'] = np.fabs((df_totalData.Uy[0] - df_totalData['Uy'])/(df_totalData.Uy[0]+small_value))*100
       df_totalData['Uz-error'] = np.fabs((df_totalData.Uz[0] - df_totalData['Uz'])/(df_totalData.Uz[0]+small_value))*100
       #print df_totalData.describe()
       return df_totalData

#   def plotting_error(self):
#       pylab.figure('error')
#       pylab.clf()
#       pylab.subplot(111)
#       ax=pylab.subplot(111)
#       pylab.ylabel(r'${\rm error [\%]}$')
#       pylab.xlabel(r'${\rm X [m]}$')
#       #ax.set_ylim(-20,20)
#       ax.set_xlim(5,5000)
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['p-error'], color='r', linewidth=line_width,label= r'$P_{error}$')
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['k-error'], color='g', linewidth=2.0,label= r'$k_{error}$')
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['nut-error'], color='b', linewidth=2.0,label= r'$\nu _{error}$')
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['epsilon-error'], color='y', linewidth=2.0,label= r'$\epsilon _{error}$')
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['Temp-error'], 'r--', linewidth=2.0,label= r'$T_{error}$')
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['Ux-error'], 'g--', linewidth=2.0,label= r'$Ux_{error}$')
#       plt.plot(self.define_newDataFile().index, self.define_newDataFile()['Uz-error'], 'b--', linewidth=2.0,label= r'$Uy_{error}$')
#
#       pylab.gcf().autofmt_xdate()
#       plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
#       plt.legend(loc='best')
#       plt.gcf().autofmt_xdate()
#       plt.savefig('error.tif',dpi=1200,bbox_inches='tight')
#       #plt.draw()
#       plt.show()       
#       #df_totalData['Temp-error'].plot(ax=axes[0,0]); axes[0,0].set_title('Temp-error');
#       #self.define_newDataFile()['p-error'].plot();
#       #plt.show()
#       return

#   def plotting_graph(self,axis_name,parameter):
#       pylab.figure('parameter')
#       pylab.clf()
#       pylab.subplot(111)
#       ax=pylab.subplot(111)
#       if (axis_name == 'x' or axis_name == 'y'):
#           pylab.ylabel(r'${\rm Par []}$')
#           pylab.xlabel(r'${\rm x [m]}$')
#           #ax.set_ylim(0,1)
#           #ax.set_xlim(-0.1,5)
#           plt.plot(self.define_newDataFile().index, self.define_newDataFile()[parameter],color='r', linewidth=2.0,label= r'$k$')
#       else:
#           pylab.ylabel(r'${\rm z [m]}$')
#           pylab.xlabel(r'${\rm Par []}$')
#           #ax.set_ylim(0,1)
#           #ax.set_xlim(-0.1,5)
#           plt.plot(self.define_newDataFile()[parameter], self.define_newDataFile().index, color='r', linewidth=2.0,label= r'$k$')
#       pylab.gcf().autofmt_xdate()
#       plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
#       plt.legend(loc='best')
#       plt.gcf().autofmt_xdate()
#       plt.savefig(axis_name + '_' + parameter+'_graph.tif',dpi=1200,bbox_inches='tight')
#       #plt.draw()
#       plt.show()       
#       #df_totalData['Temp-error'].plot(ax=axes[0,0]); axes[0,0].set_title('Temp-error');
#       #self.define_newDataFile()['p-error'].plot();
#       #plt.show()
#       return
# ------------------------------------------------------------------------------------------
# Plotting Function
def plotting_error(dfx,dfz,path):
    rho_a=1.186
    x_lable_size = 50
    y_lable_size = 50
    legend_size = 50
#    titel_size = 40
    figure_size=(20* 1.618, 20)
    line_width = 3.5
    #------------------------------------------------
    pylab.figure('error_x1',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm Error [\%]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfx.index, dfx['p_err_1'], 'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfx.index, dfx['T_err_1'], 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfx.index, dfx['k_err_1'], 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfx.index, dfx['u_err_1'], 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfx.index, dfx['nut_err_1'], 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfx.index, dfx['eps_err_1'], 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_x1.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_x2',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm Error [\%]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfx.index, dfx['p_err_2'], 'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfx.index, dfx['T_err_2'], 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfx.index, dfx['k_err_2'], 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfx.index, dfx['u_err_2'], 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfx.index, dfx['nut_err_2'], 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfx.index, dfx['eps_err_2'], 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_x2.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_x3',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm Error [\%]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfx.index, dfx['p_err_3'], 'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfx.index, dfx['T_err_3'], 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfx.index, dfx['k_err_3'], 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfx.index, dfx['u_err_3'], 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfx.index, dfx['nut_err_3'], 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfx.index, dfx['eps_err_3'], 'g-', linewidth=1.0,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_x3.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_z1',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm Error [\%]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfz['p_err_1'], dfz.index,  'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfz['T_err_1'], dfz.index, 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfz['k_err_1'], dfz.index, 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfz['u_err_1'], dfz.index, 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfz['nut_err_1'], dfz.index, 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfz['eps_err_1'], dfz.index, 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_z1.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_z2',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm Error [\%]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfz['p_err_2'], dfz.index,  'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfz['T_err_2'], dfz.index, 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfz['k_err_2'], dfz.index, 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfz['u_err_2'], dfz.index, 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfz['nut_err_2'], dfz.index, 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfz['eps_err_2'], dfz.index, 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_z2.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_z3',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm Error [\%]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfz['p_err_3'], dfz.index,  'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfz['T_err_3'], dfz.index, 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfz['k_err_3'], dfz.index, 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfz['u_err_3'], dfz.index, 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfz['nut_err_3'], dfz.index, 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfz['eps_err_3'], dfz.index, 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_z3.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_z4',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm Error [\%]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfz['p_err_4'], dfz.index,  'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfz['T_err_4'], dfz.index, 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfz['k_err_4'], dfz.index, 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfz['u_err_4'], dfz.index, 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfz['nut_err_4'], dfz.index, 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfz['eps_err_4'], dfz.index, 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_z4.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_z5',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm Error [\%]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfz['p_err_5'], dfz.index,  'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfz['T_err_5'], dfz.index, 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfz['k_err_5'], dfz.index, 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfz['u_err_5'], dfz.index, 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfz['nut_err_5'], dfz.index, 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfz['eps_err_5'], dfz.index, 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_z5.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    pylab.figure('error_z6',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm Error [\%]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    #ax.set_ylim(-20,20)
    #ax.set_xlim(0,5000)
    #plt.plot(dfz['p_err_6'], dfz.index,  'r-', linewidth=line_width,label= r'$P_{error}$',marker='')
    plt.plot(dfz['T_err_6'], dfz.index, 'b-', linewidth=line_width,label= r'$T_{error}$',marker='')
    plt.plot(dfz['k_err_6'], dfz.index, 'k-', linewidth=line_width,label= r'$k_{error}$',marker='')
    plt.plot(dfz['u_err_6'], dfz.index, 'm-', linewidth=line_width,label= r'$U_{error}$',marker='')
    #plt.plot(dfz['nut_err_6'], dfz.index, 'y-', linewidth=line_width,label= r'$\nu _{terror}$',marker='')
    plt.plot(dfz['eps_err_6'], dfz.index, 'g-', linewidth=line_width,label= r'$\varepsilon _{error}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'error_z6.tif',bbox_inches='tight')
    #plt.draw()
    plt.show()       
    #----------------------------------------------------
    return
#---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
# Plotting Function
def plotting_sample(df_x,df_z,path):
    rho_a=1.186
    x_lable_size = 50
    y_lable_size = 50
    legend_size = 50
#    titel_size = 40
    figure_size=(20* 1.618, 20)
    line_width=3.5
    #----------------------------------------------------
    pylab.figure('epsilon-z',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm \varepsilon [m^2 s^{-3}]}$',size=x_lable_size)
    #ax.relim()
    #ax.autoscale_view()
    ax.set_ylim(0,500)
    ax.set_xlim(0,0.3)
    plt.plot(df_z['epsilon_1'], df_z.index,'r-', linewidth=line_width,label= r'$inlet$',marker='')
    plt.plot(df_z['epsilon_2'], df_z.index, 'b-', linewidth=line_width,label= r'$x=100 \, m$',marker='')
    plt.plot(df_z['epsilon_3'], df_z.index, 'k-', linewidth=line_width,label= r'$x=500 \, m$',marker='')
    plt.plot(df_z['epsilon_4'], df_z.index, 'm-', linewidth=line_width,label= r'$x=1000 \, m$',marker='')
    plt.plot(df_z['epsilon_5'], df_z.index, 'g-', linewidth=line_width,label= r'$x=2500 \, m$',marker='')
    plt.plot(df_z['epsilon_6'], df_z.index, 'y-', linewidth=line_width,label= r'$Outlet$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'epsilon_z.tif',bbox_inches='tight')
    plt.draw()
   # plt.show()       
    #df_z_totalData['Temp-error'].plot(ax=axes[0,0]); axes[0,0].set_title('Temp-error');
    #self.define_newDataFile()['p-error'].plot();
    #plt.show()
    #------------------------------------------
    pylab.figure('k-z',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm k [m^2 s^{-2}]}$',size=x_lable_size)
    ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_z['k_1'], df_z.index,'r-', linewidth=line_width,label= r'$Inlet$',marker='')
    plt.plot(df_z['k_2'], df_z.index, 'b-', linewidth=line_width,label= r'$x=100 \, m$',marker='')
    plt.plot(df_z['k_3'], df_z.index, 'k-', linewidth=line_width,label= r'$x=500 \, m$',marker='')
    plt.plot(df_z['k_4'], df_z.index, 'm-', linewidth=line_width,label= r'$x=1000 \, m$',marker='')
    plt.plot(df_z['k_5'], df_z.index, 'g-', linewidth=line_width,label= r'$x=2500 \, m$',marker='')
    plt.plot(df_z['k_6'], df_z.index, 'y-', linewidth=line_width,label= r'$Outlet$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'k_z.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #----------------------------------------------
    pylab.figure('T-z',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm T [K]}$',size=x_lable_size)
    ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_z['T_1'], df_z.index,'r-', linewidth=line_width,label= r'$Inlet$',marker='')
    plt.plot(df_z['T_2'], df_z.index, 'b-', linewidth=line_width,label= r'$x=100 \, m$',marker='')
    plt.plot(df_z['T_3'], df_z.index, 'k-', linewidth=line_width,label= r'$x=500 \, m$',marker='')
    plt.plot(df_z['T_4'], df_z.index, 'm-', linewidth=line_width,label= r'$x=1000 \, m$',marker='')
    plt.plot(df_z['T_5'], df_z.index, 'g-', linewidth=line_width,label= r'$x=2500 \, m$',marker='')
    plt.plot(df_z['T_6'], df_z.index, 'y-', linewidth=line_width,label= r'$Outlet$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'T_z.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    pylab.figure('nut-z',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm nut []}$',size=x_lable_size)
    ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_z['nut_1'], df_z.index,'r-', linewidth=line_width,label= r'$Inlet$',marker='')
    plt.plot(df_z['nut_2'], df_z.index, 'b-', linewidth=line_width,label= r'$x=100 \, m$',marker='')
    plt.plot(df_z['nut_3'], df_z.index, 'k-', linewidth=line_width,label= r'$x=500 \, m$',marker='')
    plt.plot(df_z['nut_4'], df_z.index, 'm-', linewidth=line_width,label= r'$x=1000 \, m$',marker='')
    plt.plot(df_z['nut_5'], df_z.index, 'g-', linewidth=line_width,label= r'$x=2500 \, m$',marker='')
    plt.plot(df_z['nut_6'], df_z.index, 'y-', linewidth=line_width,label= r'$Outlet$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'nut_z.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    pylab.figure('P-z',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm P [K]}$',size=x_lable_size)
    ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_z['p_1']*rho_a, df_z.index,'r-', linewidth=line_width,label= r'$Inlet$',marker='')
    plt.plot(df_z['p_2']*rho_a, df_z.index, 'b-', linewidth=line_width,label= r'$x=100 \, m$',marker='')
    plt.plot(df_z['p_3']*rho_a, df_z.index, 'k-', linewidth=line_width,label= r'$x=500 \, m$',marker='')
    plt.plot(df_z['p_4']*rho_a, df_z.index, 'm-', linewidth=line_width,label= r'$x=1000 \, m$',marker='')
    plt.plot(df_z['p_5']*rho_a, df_z.index, 'g-', linewidth=line_width,label= r'$x=2500 \, m$',marker='')
    plt.plot(df_z['p_6']*rho_a, df_z.index, 'y-', linewidth=line_width,label= r'$Outlet$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'P_z.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    pylab.figure('U-z',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.ylabel(r'${\rm z [m]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm U [m/s]}$',size=x_lable_size)
    ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_z['U_0_1'], df_z.index,'r-', linewidth=line_width,label= r'$Inlet$',marker='')
    plt.plot(df_z['U_0_2'], df_z.index, 'b-', linewidth=line_width,label= r'$x=100 \, m$',marker='')
    plt.plot(df_z['U_0_3'], df_z.index, 'k-', linewidth=line_width,label= r'$x=500 \, m$',marker='')
    plt.plot(df_z['U_0_4'], df_z.index, 'm-', linewidth=line_width,label= r'$x=1000 \, m$',marker='')
    plt.plot(df_z['U_0_5'], df_z.index, 'g-', linewidth=line_width,label= r'$x=2500 \, m$',marker='')
    plt.plot(df_z['U_0_6'], df_z.index, 'y-', linewidth=line_width,label= r'$Outlet$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'U_z.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #=============================================================================
    pylab.figure('epsilon-x',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm \varepsilon [m^2 s^{-3}]}$',size=y_lable_size)
    #ax.relim()
    #ax.autoscale_view()
    #ax.set_ylim(0,500)
    #ax.set_xlim(0,0.3)
    plt.plot(df_x.index, df_x['epsilon_1'],'r-', linewidth=line_width,label= r'$\varepsilon_1$',marker='')
    plt.plot(df_x.index, df_x['epsilon_2'], 'b-', linewidth=line_width,label= r'$\varepsilon _2$',marker='')
    plt.plot(df_x.index, df_x['epsilon_3'], 'k-', linewidth=line_width,label= r'$\varepsilon _3$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'epsilon_x.tif',bbox_inches='tight')
    plt.draw()
   # plt.show()       
    #df_x_totalData['Temp-error'].plot(ax=axes[0,0]); axes[0,0].set_title('Temp-error');
    #self.define_newDataFile()['p-error'].plot();
    #plt.show()
    #------------------------------------------
    pylab.figure('k-x',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm k [m^2 s^{-2}]}$',size=y_lable_size)
    #ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_x.index, df_x['k_1'], 'r-', linewidth=line_width,label= r'$k_1$',marker='')
    plt.plot(df_x.index, df_x['k_2'], 'b-', linewidth=line_width,label= r'$k_2$',marker='')
    plt.plot(df_x.index, df_x['k_3'], 'k-', linewidth=line_width,label= r'$k_3$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'k_x.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #----------------------------------------------
    pylab.figure('T-x',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm T [K]}$',size=y_lable_size)
    #ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_x.index, df_x['T_1'],'r-', linewidth=line_width,label= r'$T_1$',marker='')
    plt.plot(df_x.index, df_x['T_2'],'b-', linewidth=line_width,label= r'$T_2$',marker='')
    plt.plot(df_x.index, df_x['T_3'],'k-', linewidth=line_width,label= r'$T_3$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'T-x.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    pylab.figure('nut-x',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm nut []}$',size=y_lable_size)
    #ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_x.index, df_x['nut_1'],'r-', linewidth=line_width,label= r'$\nu _{t1}$',marker='')
    plt.plot(df_x.index, df_x['nut_2'],'b-', linewidth=line_width,label= r'$\nu _{t2}$',marker='')
    plt.plot(df_x.index, df_x['nut_3'],  'k-', linewidth=line_width,label= r'$\nu _{t3}$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'nut_x.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    pylab.figure('P-x',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm P [K]}$',size=y_lable_size)
    #ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_x.index, df_x['p_1']*rho_a, 'r-', linewidth=line_width,label= r'$P_1$',marker='')
    plt.plot(df_x.index, df_x['p_2']*rho_a,  'b-', linewidth=line_width,label= r'$P_2$',marker='')
    plt.plot(df_x.index, df_x['p_3']*rho_a,  'k-', linewidth=line_width,label= r'$P_3$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'P_x.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    pylab.figure('U-x',figsize=figure_size)
    pylab.clf()
    pylab.subplot(111)
    ax=pylab.subplot(111)
    pylab.xlabel(r'${\rm X [m]}$',size=x_lable_size)
    pylab.ylabel(r'${\rm U [m/s]}$',size=y_lable_size)
    #ax.set_ylim(0,500)
    #ax.set_xlim(0,0.5)
    plt.plot(df_x.index, df_x['U_0_1'], 'r-', linewidth=line_width,label= r'$P_1$',marker='')
    plt.plot(df_x.index, df_x['U_0_2'], 'b-', linewidth=line_width,label= r'$P_2$',marker='')
    plt.plot(df_x.index, df_x['U_0_3'], 'k-', linewidth=line_width,label= r'$P_3$',marker='')
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(path+'U_x.tif',bbox_inches='tight')
    plt.draw()
    plt.show()       
    #-----------------------------------------------
    return
#--------------------------------------------------------------------------
# Definig new data set for each sample file
path_sampleFiles = "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/"


#unstable_ABL_kEpsilon_class = reading_sample(path_sampleFiles + "unstable_ABL_kEpsilon/5912/x2_T_epsilon_k_nut_p.csv", \
#                                           path_sampleFiles + "unstable_ABL_kEpsilon/5912/x2_U.csv" )
#
#unstable_ABL_class = reading_sample(path_sampleFiles + "unstable_ABL/2700/x2_T_epsilon_k_nut_p.csv", \
#                                           path_sampleFiles + "unstable_ABL/2700/x2_U.csv" )
#                                           
#stable_ABL_kEpsilon_class = reading_sample(path_sampleFiles + "stable_ABL_kEpsilon/1783.7/x2_T_epsilon_k_nut_p.csv", \
#                                           path_sampleFiles + "stable_ABL_kEpsilon/1783.7/x2_U.csv" )
#
#
#stable_ABL_kEpsilon_Cb_02_class = reading_sample(path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x2_T_epsilon_k_nut_p.csv", \
#                                           path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x2_U.csv" )
#
#stable_ABL_kEpsilon_class = reading_sample(path_sampleFiles + "stable_ABL_kEpsilon/1783.7/x2_T_epsilon_k_nut_p.csv", \
#                                           path_sampleFiles + "stable_ABL_kEpsilon/1783.7/x2_U.csv" )
#
#neutral_ABL_class = reading_sample(path_sampleFiles + "neutral_ABL/2077.7/x2_T_epsilon_k_nut_p.csv", \
#                                           path_sampleFiles + "neutral_ABL/2077.7/x2_U.csv" )
#
## definig the dataFrame to use in plotting!                                           
#unstable_ABL_kEpsilon = unstable_ABL_kEpsilon_class.define_newDataFile()
#unstable_ABL = unstable_ABL_class.define_newDataFile()
#
#stable_ABL_kEpsilon_Cb_02 = stable_ABL_kEpsilon_Cb_02_class.define_newDataFile()
#stable_ABL_kEpsilon = stable_ABL_kEpsilon_class.define_newDataFile()
#neutral_ABL = neutral_ABL_class.define_newDataFile()

# --------------------------------------------------------------------------------------------------------------

#stable_ABL_kEpsilon_Cb_z_class = reading_z_sample("stable","z",path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z1_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z1_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z2_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z2_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z3_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z3_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z4_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z4_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z5_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z5_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z6_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/z6_U.csv"
#                                           )
#stable_ABL_kEpsilon_Cb_z = stable_ABL_kEpsilon_Cb_z_class.reading_CSV()
#
#stable_ABL_kEpsilon_Cb_x_class = reading_x_sample("stable","x",path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/x1_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/x1_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/x2_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/x2_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/x3_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb/213100/x3_U.csv", \
#                                           )
#stable_ABL_kEpsilon_Cb_x = stable_ABL_kEpsilon_Cb_x_class.reading_CSV()
##-----------------------------------------------------------------------------------------------------------
#stable_ABL_kEpsilon_Cb_02_z_class = reading_z_sample("stable","z",path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z1_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z1_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z2_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z2_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z3_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z3_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z4_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z4_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z5_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z5_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z6_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/z6_U.csv"
#                                           )
#stable_ABL_kEpsilon_Cb_02_z = stable_ABL_kEpsilon_Cb_02_z_class.reading_CSV()
#
#stable_ABL_kEpsilon_Cb_02_x_class = reading_x_sample("stable","x",path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x1_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x1_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x2_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x2_U.csv", \
#                                        path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x3_T_epsilon_k_nut_p.csv", path_sampleFiles + "stable_ABL_kEpsilon_Cb_02/3798.3/x3_U.csv", \
#                                           )
#stable_ABL_kEpsilon_Cb_02_x = stable_ABL_kEpsilon_Cb_02_x_class.reading_CSV()
##-----------------------------------------------------------------------------------------------------------
#unstable_ABL_kEpsilon_Cb_z_class = reading_z_sample("unstable","z",path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z3_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z4_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z4_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z5_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z5_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z6_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/z6_U.csv"
#                                           )
#unstable_ABL_kEpsilon_Cb_z = unstable_ABL_kEpsilon_Cb_z_class.reading_CSV()
#
#unstable_ABL_kEpsilon_Cb_x_class = reading_x_sample("unstable","x",path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/x1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/x1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/x2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/x2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/x3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb/226435.7/x3_U.csv", \
#                                           )
#unstable_ABL_kEpsilon_Cb_x = unstable_ABL_kEpsilon_Cb_x_class.reading_CSV()
##-----------------------------------------------------------------------------------------------------------
#unstable_ABL_kEpsilon_Cb_02_z_class = reading_z_sample("unstable","z",path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z3_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z4_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z4_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z5_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z5_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z6_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/z6_U.csv"
#                                           )
#unstable_ABL_kEpsilon_Cb_02_z = unstable_ABL_kEpsilon_Cb_02_z_class.reading_CSV()
#
#unstable_ABL_kEpsilon_Cb_02_x_class = reading_x_sample("unstable","x",path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/x1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/x1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/x2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/x2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/x3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_02/230192.9/x3_U.csv", \
#                                           )
#unstable_ABL_kEpsilon_Cb_02_x = unstable_ABL_kEpsilon_Cb_02_x_class.reading_CSV()
##-----------------------------------------------------------------------------------------------------------
#unstable_ABL_kEpsilon_Cb_03_z_class = reading_z_sample("unstable","z",path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z3_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z4_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z4_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z5_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z5_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z6_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/z6_U.csv"
#                                           )
#unstable_ABL_kEpsilon_Cb_03_z = unstable_ABL_kEpsilon_Cb_03_z_class.reading_CSV()
#
#unstable_ABL_kEpsilon_Cb_03_x_class = reading_x_sample("unstable","x",path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x3_U.csv", \
#                                           )
#unstable_ABL_kEpsilon_Cb_03_x = unstable_ABL_kEpsilon_Cb_03_x_class.reading_CSV()

#-----------------------------------------------------------------------------------------------------------
#unstable_ABL_kEpsilon_buoyancy_lapse_z_class = reading_z_sample("unstable","z",path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z3_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z4_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z4_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z5_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z5_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z6_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/z6_U.csv"
#                                           )
#unstable_ABL_kEpsilon_buoyancy_lapse_z = unstable_ABL_kEpsilon_buoyancy_lapse_z_class.reading_CSV()
#
#unstable_ABL_kEpsilon_buoyancy_lapse_x_class = reading_x_sample("unstable","x",path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x1_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_Cb_03/3363.6/x1_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/x2_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/x2_U.csv", \
#                                        path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/x3_T_epsilon_k_nut_p.csv", path_sampleFiles + "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/x3_U.csv", \
#                                           )
#unstable_ABL_kEpsilon_buoyancy_lapse_x = unstable_ABL_kEpsilon_buoyancy_lapse_x_class.reading_CSV()

# Plotting Results

#--------------------------------------------------------------------------------------
#plotting_error(unstable_ABL_kEpsilon,unstable_ABL)


#
#plotting_sample(stable_ABL_kEpsilon_Cb_x, stable_ABL_kEpsilon_Cb_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_Cb_")
#plotting_error(stable_ABL_kEpsilon_Cb_x, stable_ABL_kEpsilon_Cb_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_Cb_")
#
#plotting_sample(stable_ABL_kEpsilon_Cb_02_x, stable_ABL_kEpsilon_Cb_02_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_Cb_02_")
#plotting_error(stable_ABL_kEpsilon_Cb_02_x, stable_ABL_kEpsilon_Cb_02_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_Cb_02_")
#
#
#plotting_sample(unstable_ABL_kEpsilon_Cb_x, unstable_ABL_kEpsilon_Cb_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_Cb_")
#plotting_error(unstable_ABL_kEpsilon_Cb_x, unstable_ABL_kEpsilon_Cb_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_Cb_")
#
#plotting_sample(unstable_ABL_kEpsilon_Cb_02_x, unstable_ABL_kEpsilon_Cb_02_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_Cb_02_")
#plotting_error(unstable_ABL_kEpsilon_Cb_02_x, unstable_ABL_kEpsilon_Cb_02_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_Cb_02_")
#
#plotting_sample(unstable_ABL_kEpsilon_Cb_03_x, unstable_ABL_kEpsilon_Cb_03_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_Cb_03_")
#plotting_error(unstable_ABL_kEpsilon_Cb_03_x, unstable_ABL_kEpsilon_Cb_03_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_Cb_03_")

#plotting_sample(unstable_ABL_kEpsilon_buoyancy_lapse_x, unstable_ABL_kEpsilon_buoyancy_lapse_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_buoyancy_lapse_")
#plotting_error(unstable_ABL_kEpsilon_buoyancy_lapse_x, unstable_ABL_kEpsilon_buoyancy_lapse_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_buoyancy_lapse_").
#******************************************************************************************************************************************
#************************************************************************************************************************************
#path_sampleFiles = "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/"
#path= "neutral_ABL_mod/2034.6/"
#neutral_ABL_mod_z_class = reading_z_sample("neutral","z",path_sampleFiles + path + "z1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z1_U.csv", \
#                                        path_sampleFiles + path + "z2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z2_U.csv", \
#                                        path_sampleFiles + path + "z3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z3_U.csv", \
#                                        path_sampleFiles + path + "z4_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z4_U.csv", \
#                                        path_sampleFiles + path + "z5_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z5_U.csv", \
#                                        path_sampleFiles + path + "z6_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z6_U.csv"
#                                           )
#neutral_ABL_mod_z = neutral_ABL_mod_z_class.reading_CSV()
#
#neutral_ABL_mod_x_class = reading_x_sample("neutral","x",path_sampleFiles+ path + "x1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x1_U.csv", \
#                                        path_sampleFiles + path + "x2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x2_U.csv", \
#                                        path_sampleFiles + path + "x3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x3_U.csv", \
#                                           )
#neutral_ABL_mod_x = neutral_ABL_mod_x_class.reading_CSV()
# ---------------------------------------------------------------------------------------------------------
path_sampleFiles = "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/"
path= "neutral_ABL_mod_02/52850/"
neutral_ABL_mod_02_z_class = reading_z_sample("neutral","z",path_sampleFiles + path + "z1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z1_U.csv", \
                                        path_sampleFiles + path + "z2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z2_U.csv", \
                                        path_sampleFiles + path + "z3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z3_U.csv", \
                                        path_sampleFiles + path + "z4_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z4_U.csv", \
                                        path_sampleFiles + path + "z5_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z5_U.csv", \
                                        path_sampleFiles + path + "z6_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z6_U.csv"
                                           )
neutral_ABL_mod_02_z = neutral_ABL_mod_02_z_class.reading_CSV()

neutral_ABL_mod_02_x_class = reading_x_sample("neutral","x",path_sampleFiles+ path + "x1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x1_U.csv", \
                                        path_sampleFiles + path + "x2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x2_U.csv", \
                                        path_sampleFiles + path + "x3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x3_U.csv", \
                                           )
neutral_ABL_mod_02_x = neutral_ABL_mod_02_x_class.reading_CSV()
# ---------------------------------------------------------------------------------------------------------
#path= "stable_ABL_kEpsilon_buoyancy_lapse_g/2743.7/"
#stable_ABL_kEpsilon_buoyancy_lapse_g_z_class = reading_z_sample("stable","z",path_sampleFiles + path + "z1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z1_U.csv", \
#                                        path_sampleFiles + path + "z2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z2_U.csv", \
#                                        path_sampleFiles + path + "z3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z3_U.csv", \
#                                        path_sampleFiles + path + "z4_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z4_U.csv", \
#                                        path_sampleFiles + path + "z5_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z5_U.csv", \
#                                        path_sampleFiles + path + "z6_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z6_U.csv"
#                                           )
#stable_ABL_kEpsilon_buoyancy_lapse_g_z = stable_ABL_kEpsilon_buoyancy_lapse_g_z_class.reading_CSV()
#
#stable_ABL_kEpsilon_buoyancy_lapse_g_x_class = reading_x_sample("stable","x",path_sampleFiles + path + "x1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x1_U.csv", \
#                                        path_sampleFiles + path + "x2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x2_U.csv", \
#                                        path_sampleFiles + path + "x3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x3_U.csv", \
#                                           )
#stable_ABL_kEpsilon_buoyancy_lapse_g_x = stable_ABL_kEpsilon_buoyancy_lapse_g_x_class.reading_CSV()
##print neutral_ABL_z.describe()
#---------- --------------------------------------------------------------------------------------
path= "stable_ABL_kEpsilon_buoyancy_lapse_g_mod/2336.8/"
stable_ABL_kEpsilon_buoyancy_lapse_g_mod_z_class = reading_z_sample("stable","z",path_sampleFiles + path + "z1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z1_U.csv", \
                                        path_sampleFiles + path + "z2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z2_U.csv", \
                                        path_sampleFiles + path + "z3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z3_U.csv", \
                                        path_sampleFiles + path + "z4_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z4_U.csv", \
                                        path_sampleFiles + path + "z5_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z5_U.csv", \
                                        path_sampleFiles + path + "z6_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z6_U.csv"
                                           )
stable_ABL_kEpsilon_buoyancy_lapse_g_mod_z = stable_ABL_kEpsilon_buoyancy_lapse_g_mod_z_class.reading_CSV()

stable_ABL_kEpsilon_buoyancy_lapse_g_mod_x_class = reading_x_sample("stable","x",path_sampleFiles + path + "x1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x1_U.csv", \
                                        path_sampleFiles + path + "x2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x2_U.csv", \
                                        path_sampleFiles + path + "x3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x3_U.csv", \
                                           )
stable_ABL_kEpsilon_buoyancy_lapse_g_mod_x = stable_ABL_kEpsilon_buoyancy_lapse_g_mod_x_class.reading_CSV()
#print neutral_ABL_z.describe()
#-------------------------------------------------------------------------------------
#path= "unstable_ABL_kEpsilon_buoyancy_lapse/129.5/"
#unstable_ABL_kEpsilon_buoyancy_lapse_g_z_class = reading_z_sample("unstable","z",path_sampleFiles + path + "z1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z1_U.csv", \
#                                        path_sampleFiles + path + "z2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z2_U.csv", \
#                                        path_sampleFiles + path + "z3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z3_U.csv", \
#                                        path_sampleFiles + path + "z4_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z4_U.csv", \
#                                        path_sampleFiles + path + "z5_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z5_U.csv", \
#                                        path_sampleFiles + path + "z6_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z6_U.csv"
#                                           )
#unstable_ABL_kEpsilon_buoyancy_lapse_g_z = unstable_ABL_kEpsilon_buoyancy_lapse_g_z_class.reading_CSV()
#
#unstable_ABL_kEpsilon_buoyancy_lapse_g_x_class = reading_x_sample("unstable","x",path_sampleFiles + path + "x1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x1_U.csv", \
#                                        path_sampleFiles + path + "x2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x2_U.csv", \
#                                        path_sampleFiles + path + "x3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x3_U.csv", \
#                                           )
#unstable_ABL_kEpsilon_buoyancy_lapse_g_x = unstable_ABL_kEpsilon_buoyancy_lapse_g_x_class.reading_CSV()
##print neutral_ABL_z.describe()
#-------------------------------------------------------------------------------------
path= "unstable_ABL_kEpsilon_bupyancy_lapse_g_mod/629.7/"
unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_z_class = reading_z_sample("unstable","z",path_sampleFiles + path + "z1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z1_U.csv", \
                                        path_sampleFiles + path + "z2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z2_U.csv", \
                                        path_sampleFiles + path + "z3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z3_U.csv", \
                                        path_sampleFiles + path + "z4_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z4_U.csv", \
                                        path_sampleFiles + path + "z5_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z5_U.csv", \
                                        path_sampleFiles + path + "z6_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "z6_U.csv"
                                           )
unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_z = unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_z_class.reading_CSV()

unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_x_class = reading_x_sample("unstable","x",path_sampleFiles + path + "x1_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x1_U.csv", \
                                        path_sampleFiles + path + "x2_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x2_U.csv", \
                                        path_sampleFiles + path + "x3_T_epsilon_k_nut_p.csv", path_sampleFiles + path + "x3_U.csv", \
                                           )
unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_x = unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_x_class.reading_CSV()
#print neutral_ABL_z.describe()
#******************************************************************************************************************
#plotting_sample(neutral_ABL_mod_x, neutral_ABL_mod_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/neutral_ABL_mod_")
#plotting_error(neutral_ABL_mod_x, neutral_ABL_mod_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/neutral_ABL_mod_")
#
plotting_sample(neutral_ABL_mod_02_x, neutral_ABL_mod_02_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/neutral_ABL_mod_02_")
plotting_error(neutral_ABL_mod_02_x, neutral_ABL_mod_02_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/neutral_ABL_mod_02_")

#plotting_sample(unstable_ABL_kEpsilon_buoyancy_lapse_g_x, unstable_ABL_kEpsilon_buoyancy_lapse_g_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_buoyancy_lapse_g_")
#plotting_error(unstable_ABL_kEpsilon_buoyancy_lapse_g_x, unstable_ABL_kEpsilon_buoyancy_lapse_g_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_buoyancy_lapse_g_")

plotting_sample(unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_x, unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_")
plotting_error(unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_x, unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/unstable_ABL_kEpsilon_buoyancy_lapse_g_mod_")

#plotting_sample(stable_ABL_kEpsilon_buoyancy_lapse_g_x, stable_ABL_kEpsilon_buoyancy_lapse_g_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_buoyancy_lapse_g_")
#plotting_error(stable_ABL_kEpsilon_buoyancy_lapse_g_x, stable_ABL_kEpsilon_buoyancy_lapse_g_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_buoyancy_lapse_g_")

plotting_sample(stable_ABL_kEpsilon_buoyancy_lapse_g_mod_x, stable_ABL_kEpsilon_buoyancy_lapse_g_mod_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_buoyancy_lapse_g_mod_")
plotting_error(stable_ABL_kEpsilon_buoyancy_lapse_g_mod_x, stable_ABL_kEpsilon_buoyancy_lapse_g_mod_z, "/media/localadmin/Seagate_4T/Validation_ABLthermo/analysing_results/stable_ABL_kEpsilon_buoyancy_lapse_g_mod_")
#******************************************************************************************************



