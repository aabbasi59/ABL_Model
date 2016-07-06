# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:23:16 2012
@author: aliabbasi
"""
import pdb
import meteolib
import os    # operating system routines
import scipy # Scientific Python functions
import pylab # Plotting functions
import math
import numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd  
import glob  
import matplotlib 
import matplotlib.dates as mdates
from matplotlib import rc, rcParams
from datetime import datetime, timedelta
import reading_input
import totalLibs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#----------------------------------------------------------------------------------------
def writeData_textFile_ABLModel(outfile, idpar=scipy.array([]),y1=scipy.array([])):
    '''
    Function to write data in a text file "filename" with values being 
    separated by a "TAB" according to the acceptable format to read in OpenFOAM

    Usage:
        writeData(filename,par1,par2)
    '''   
    # Open and read the file
    outputfile = open(outfile, 'w');
    outputfile.write ('(\n')
    #outputfile.write ('#time, BC\n')
    # Start writing the data into the file
    for i in range(0,len(idpar)):
        filelist=[]
        filelist.append('(')
        filelist.append(str('%8.1f' % float((idpar[i]))))
        filelist.append('\t')
        filelist.append(str('%8.5f' % float(y1[i])))
        filelist.append(')')
        #filelist.append('\t')
        #filelist.append(str('%5.2f' % float(y2[i])))
        filelist.append('\n')
        outputfile.writelines(filelist)
    outputfile.write (')\n')    
    outputfile.close()
    
    return()
#-----------------------------------------------------------------------------------
pylab.rc('text', usetex=True)
# Change default settings for figures
newdefaults = {'fontname':    'serif',  # Use Arial font
               'backend':       'svg',  # Save figure as EPS file   
               'axes.labelsize':   60,  # Axis label size in points
               'text.fontsize':    60,  # Text size in points
               'legend.fontsize':  60,  # Legend label size in points
               'xtick.labelsize':  60,  # x-tick label size in points
               'ytick.labelsize':  60,  # y-tick label size in points
               'lines.markersize': 60,  # markersize, in points
               'lines.linewidth':   1.5 # line width in points
               }
pylab.rcParams.update(newdefaults)
#-------------------------------------------------------
#To felexible time scale the different DatFrame are defined here.
#df_original = reading_input.df_totalData
#df_hourly = reading_input.df_totalData_hourlyAve
#df_daily = reading_input.df_totalData_dailyAve
#df_weekly = reading_input.df_totalData_weeklyAve
#df_monthly = reading_input.df_totalData_monthlyAve
# this DataFrame will be analysed. It should be changed for your desired data set.
#df_analysed = df_hourly
#-----------------------------------------------------------------------------------
start_Date = '12/1/2012 23:59:00'# '12/15/2012 23:59:54'
end_Date = '12/3/2012 23:59:53' # '12/20/2012 23:59:53'
end_Date_plot = '12/3/2012 23:59:53' #  '12/20/2012 23:59:53'
# generating a new DataFrame according to the desired range from start to end date

df_original_selPeriod = reading_input.df_totalData[(reading_input.df_totalData.index >= start_Date) & (reading_input.df_totalData.index <= end_Date)]
df_hourly_selPeriod = reading_input.df_totalData_hourlyAve[(reading_input.df_totalData_hourlyAve.index >= start_Date) & (reading_input.df_totalData_hourlyAve.index <= end_Date)]
df_analysed = df_original_selPeriod #df_hourly_selPeriod
# Putting the values from totalData.csv into the arrays
dec_time =  df_analysed.index
airtemp = df_analysed['air_Temp']              # [°C] Air temperature
surftemp = df_analysed['surf_Temp']            # [°C]  Water surface temperature
Rs = df_analysed['Rs']                         # [W/m2] Short-wave radiation
RH = df_analysed['RH']                         # [%]
U = df_analysed['U2']                          # [m/s]
D = df_analysed['windDir']                     # [°N] from the north in clock-wise direction
H_m = df_analysed['senHeat_m']              # measured sensible heat flux
airpress = df_analysed['air_pressure_m']       # Air pressure in [pa]
airdensity_m = df_analysed['air_Density_m']      # measured air density kg/m-3
nu =  df_analysed['air_nu']
Lambda = df_analysed['Lambda']
cp = df_analysed['air_Cp']
qs = df_analysed['X_s']
qz = df_analysed['X_a']
es = df_analysed['es']
ea = df_analysed['ea']
e_sat = df_analysed['esat']
virt_airtemp = df_analysed['virt_air_Temp']
virt_surftemp = df_analysed['virt_surf_Temp']
gamma = df_analysed['gamma'] 
u_Star_m = df_analysed['u_Star_m']
kessi_m = df_analysed['kessi_m']
#---------------------------------------------------------------------------------------
# Defining new lists for new parameters
OB_L = []
SenHeat = []
LatHeat = []
vapor_stab_function = []
momentum_stab_function = []
u_Star = []
E = []
C_Momentum = []
C_Heat =[]
C_Vapor = []
C_Momentum_N = []
C_Heat_N =[]
C_Vapor_N = []
kesi = []
z_momentum = []
z_heat = []
z_vapor =[]
U_fc =[]
CD_CDN=[]
CH_CHN=[]
CE_CEN=[]
delta_virtemp=[]
delta_temp=[]
#-------------------------------------------------------------------------------

# Main routine of program
#pdb.set_trace()
data_size = scipy.size(df_analysed.index)
#print airtemp[0:10]
# Put the minimum wind speed equal to 0.1 [m/s]
for i in range(0, data_size):
    if (U[i]>=0.1):
        U[i]=U[i]
    else:
        U[i]=0.1
# Main Part of Code
for i in range(0, data_size):
    if (U[i]>=0.1):
        uStar_initial = meteolib.windZ1toZ2(reading_input.ZZ,U[i],10.00)*math.pow(0.0015*math.pow(1.00+math.exp((-meteolib.windZ1toZ2(reading_input.ZZ,U[i],10.00)+12.50)/1.56),-1.00)+0.00104,0.5)
        z00 = totalLibs.z0(uStar_initial,nu[i])
        #uStar = meteolib.windZ1toZ2withz0(reading_input.ZZ,U[i],10.00,z00)*math.pow(0.0015*math.pow(1.00+ \
        #        math.exp((-meteolib.windZ1toZ2withz0(reading_input.ZZ,U[i],10.00,z00)+12.50)/1.56),-1.00)+0.00104,0.5)
        uStar = totalLibs.uStar_calc(reading_input.van_karman,U[i],reading_input.ZZ,z00)
        z01 = totalLibs.z0(uStar,nu[i])
        while  (abs(z01-z00)>(reading_input.error_criteria*z00)):
            #uStar = meteolib.windZ1toZ2withz0(reading_input.ZZ,U[i],10.00,z01)*math.pow(0.0015*math.pow(1.00+ \
            #        math.exp((-meteolib.windZ1toZ2withz0(reading_input.ZZ,U[i],10.00,z00)+12.50)/1.56),-1.00)+0.00104,0.5)
            uStar = totalLibs.uStar_calc(reading_input.van_karman,U[i],reading_input.ZZ,z01)
            z00=z01
            z01 = totalLibs.z0(uStar,nu[i])
        z0m = z01 
        uStar=uStar
#        C_DN = math.pow((uStar/U[i]),2.00) # Natural transfer coefficient for Momentum
        #C_DN = 0.0044*math.pow(meteolib.windZ1toZ2(reading_input.ZZ,U[i],10.00),-1.15)
        C_DN = math.pow(reading_input.van_karman/(math.log(reading_input.ZZ/z0m)),2.00) # Natural transfer coefficient for Momentum
        rough_Reynold_no = uStar*z0m / nu[i]# Roughness Reynolds Number
        z0e = z0m*math.exp(-2.67*math.pow(rough_Reynold_no,0.25)+2.57)  # Roughness length for vapor [m]
        z0h = z0e # Roughness length for heat [m]
        C_EN = reading_input.van_karman * math.pow(C_DN,0.5)/math.log(reading_input.ZZ/z0e)  # Natural transfer coefficient for vapor
        C_HN = C_EN  # Natural transfer coefficient for haet
        C_Momentum_N.append(float(C_DN))
        C_Heat_N.append(float(C_HN))
        C_Vapor_N.append(float(C_EN))
        #print C_HN
        H = airdensity_m[i]*cp[i]*C_HN*U[i]*(surftemp[i]-airtemp[i])
        #print H
        E = airdensity_m[i]*Lambda[i]*C_EN*U[i]*(qs[i]-qz[i])
        #print E
        L00 = (-airdensity_m[i]*math.pow(uStar,3)*virt_airtemp[i])/(reading_input.van_karman*reading_input.gravityAccel*(H/cp[i]+0.61*(airtemp[i]+273.16)*E/Lambda[i]))
        kesi00 = reading_input.ZZ/L00 #Stability parameter
        L01 = L00+10 #for starting the loop
        kessi01 = reading_input.ZZ/L01
        #print L01, L00, kesi00
        while ((math.fabs(math.fabs(L01)-math.fabs(L00)))>(reading_input.error_criteria*math.fabs(L00))):
            #L00 = L01
            U_fc0 =U[i]
            C_D = math.pow(reading_input.van_karman,2)/math.pow((math.log(reading_input.ZZ/z0m)-totalLibs.stability_correct_mom(kesi00)),2)
            C_E = reading_input.van_karman*math.pow(C_D,0.5)/(math.log(reading_input.ZZ/z0e)-totalLibs.stability_correct_vapor(kesi00))
            C_H = C_E 
            H = airdensity_m[i]*cp[i]*C_H*U[i]*(surftemp[i]-airtemp[i])
            E = airdensity_m[i]*Lambda[i]*C_E*U[i]*(qs[i]-qz[i])
            uStar = math.pow((C_D*math.pow(U[i],2)),0.5)
            z0m = totalLibs.z0(uStar,nu[i])
            rough_Reynold_no = uStar*z0m / nu[i]  # Roughness Reynolds Number
            z0e = z0m*math.exp(-2.67*math.pow(rough_Reynold_no,0.25)+2.57)  # Roughness length for vapor [m]
            z0h = z0e # Roughness length for heat [m]
            L00 = L01
            kessi00 = kessi01
            L01 = (-airdensity_m[i]*math.pow(uStar,3)*virt_airtemp[i])/(reading_input.van_karman*reading_input.gravityAccel*(H/cp[i]+0.61*(airtemp[i]+273.16)*E/Lambda[i]))
            kesi01 = reading_input.ZZ/L01 #Stability parameter
            #print i,kesi01,U[i]
            #gradU0 = C_D*airdensity_m[i]*math.pow(U[i],2)/(nu[i]*airdensity_m[i])
        OB_L.append(float(L01))
        SenHeat.append(float(H))
        LatHeat.append(float(E))
        #vapor_stab_function.append(float(vapor_stab_function0))
        #momentum_stab_function.append(float(momentum_stab_function0))
        u_Star.append(float(uStar))
        #E_A.append(float(E_A0))
        C_Momentum.append(float(C_D))
        C_Heat.append(float(C_H))
        C_Vapor.append(float(C_E))
        kesi.append(float(kesi01))
        z_momentum.append(float(z0m))
        z_heat.append(float(z0h))
        z_vapor.append(float(z0e)) 
        U_fc.append(float(U_fc0))
        #gradU.append(float(gradU0))
#    else:
#        #print 'qqq'
#        uStar_initial = meteolib.windZ1toZ2(reading_input.ZZ,U[i],10.00)*math.pow(0.0015*math.pow(1.00+ \
#                        math.exp((-meteolib.windZ1toZ2(reading_input.ZZ,U[i],10.00)+12.50)/1.56),-1.00)+0.00104,0.5)
#        wStar_initial = 0.2 #assumption???
#        z00 = totalLibs.z0_imp(uStar_initial,wStar_initial,nu[i])
#        #print z00
#        U_fc0 = math.sqrt(math.pow(U[i],2)+math.pow((beta*wStar_initial),2))
#        uStar = meteolib.windZ1toZ2withz0(reading_input.ZZ,U_fc0,10.00,z00)*math.pow(0.0015*math.pow(1.00+ \
#                math.exp((-meteolib.windZ1toZ2withz0(reading_input.ZZ,U_fc0,10.00,z00)+12.50)/1.56),-1.00)+0.00104,0.5)
#        #z01 = totalLibs.z0_imp(uStar,wStar_initial,nu[i])
#        z01 = totalLibs.z0_imp(uStar,0,nu[i])
#        #print z01
#        z0m = z01 
#        uStar=uStar
#        C_DN0 = math.pow(van_karman/(math.log(reading_input.ZZ/z0m)),2.00) # Natural transfer coefficient for Momentum
#        rough_Reynold_no = uStar*z0m / nu[i]# Roughness Reynolds Number
#        z0e = z0m*math.exp(-2.67*math.pow(rough_Reynold_no,0.25)+2.57)  # Roughness length for vapor [m]
#        z0h = z0e # Roughness length for heat [m]
#        C_EN0 = van_karman * math.pow(C_DN0,0.5)/math.log(reading_input.ZZ/z0e)  # Natural transfer coefficient for vapor
#        C_HN0 = C_EN0  # Natural transfer coefficient for haet
#        C_Momentum_N.append(float(C_DN0))
#        C_Heat_N.append(float(C_HN0))
#        C_Vapor_N.append(float(C_EN0))
#        H0 = airdensity_m[i]*cp[i]*C_HN0*U_fc0*(surftemp[i]-airtemp[i])
#        E0 = airdensity_m[i]*Lambda[i]*C_EN0*U_fc0*(qs[i]-qz[i])
#        L00 = (-airdensity_m[i]*math.pow(uStar,3)*virt_airtemp[i])/(van_karman*gravityAccel* \
#              (H0/cp[i]+0.61*(airtemp[i]+273.16)*E0/Lambda[i]))
#        kesi00 = reading_input.ZZ/L00 #Stability parameter
#        L01 = L00*1.5 #for starting the loop
#        kessi01 = reading_input.ZZ/L01
#              
#        while ((math.fabs(math.fabs(L01)-math.fabs(L00)))>(error_criteria)):
#            #((math.fabs(math.fabs(L01)-math.fabs(L00)))>(error_criteria*math.fabs(L00))):
#            wStar = math.pow((gravityAccel/airtemp[i]*H0*z_i),1/3)
#            U_fc0 = math.sqrt(math.pow(U[i],2)+math.pow((beta*wStar),2))
#            uStar = van_karman*beta*wStar/(math.log(reading_input.ZZ/z00)-totalLibs.stability_correct_mom(reading_input.ZZ/L01)+ \
#                    totalLibs.stability_correct_mom(z00/L01))
#            z00 = totalLibs.z0_imp(uStar,0,nu[i])
#            z0m = z00
#            C_D = math.pow(van_karman,2)/math.pow((math.log(reading_input.ZZ/z0m)-totalLibs.stability_correct_mom(kesi00)),2)
#            C_E = van_karman*math.pow(C_D,0.5)/(math.log(reading_input.ZZ/z0e)-totalLibs.stability_correct_vapor(kesi00))
#            C_H = C_E 
#            H0 = airdensity_m[i]*cp[i]*C_H*U_fc0*(surftemp[i]-airtemp[i])
#            E0 = airdensity_m[i]*Lambda[i]*C_E*U_fc0*(qs[i]-qz[i])
#            #uStar = math.pow((C_D*math.pow(U[i],2)),0.5)
#            #z0m = totalLibs.z0uStar,nu[i])
#            rough_Reynold_no = uStar*z0m / nu[i]  # Roughness Reynolds Number
#            z0e = z0m*math.exp(-2.67*math.pow(rough_Reynold_no,0.25)+2.57)  # Roughness length for vapor [m]
#            z0h = z0e # Roughness length for heat [m]
#            L00 = L01
#            kessi00 = kessi01
#            L01 = (-airdensity_m[i]*math.pow(uStar,3)*virt_airtemp[i])/(van_karman*gravityAccel* \
#                  (H0/cp[i]+0.61*(airtemp[i]+273.16)*E0/Lambda[i]))
#            kesi01 = reading_input.ZZ/L01 #Stability parameter
#            #print L01,L00, error_criteria
#            #gradU0 = C_D*airdensity_m[i]*math.pow(U[i],2)/(nu[i]*airdensity_m[i])
#        OB_L.append(float(L01))
#        SenHeat.append(float(H0))
#        LatHeat.append(float(E0))
#        #vapor_stab_function.append(float(vapor_stab_function0))
#        #momentum_stab_function.append(float(momentum_stab_function0))
#        u_Star.append(float(uStar))
#        #E_A.append(float(E_A0))
#        C_Momentum.append(float(C_D))
#        C_Heat.append(float(C_H))
#        C_Vapor.append(float(C_E))
#        kesi.append(float(kesi01))
#        z_momentum.append(float(z0m))
#        z_heat.append(float(z0h))
#        z_vapor.append(float(z0e)) 
#        U_fc.append(float(U_fc0))
#        #print i,OB_L[i],z_momentum[i]
#        #gradU.append(float(gradU0))
LatHeat_N=[]
SenHeat_N=[]
LatHeat_Rel=[]
SenHeat_Rel=[]
SenHeat_LatHeat_N=[]
SenHeat_LatHeat=[]
beta_betaN=[]
LatHeat_mmday=[]
freq_kesi_percent =[]

for i in range(0, data_size):
    LatHeat_N0=airdensity_m[i]*Lambda[i]*C_Vapor_N[i] *U[i]*(qs[i]-qz[i])
    LatHeat_N.append(float(LatHeat_N0))
    LatHeat_Rel0= LatHeat[i]/LatHeat_N[i]
    LatHeat_Rel.append(float(LatHeat_Rel0))
    SenHeat_N0=airdensity_m[i]*cp[i]*C_Heat_N[i]*U[i]*(surftemp[i]-airtemp[i])
    SenHeat_N.append(float(SenHeat_N0))
    SenHeat_Rel0= SenHeat[i]/(SenHeat_N[i]+0.001)
    SenHeat_Rel.append(float(SenHeat_Rel0))
    CD_CDN0=C_Momentum[i]/C_Momentum_N[i]
    CD_CDN.append(float(CD_CDN0))
    CH_CHN0=C_Heat[i]/C_Heat_N[i]
    CH_CHN.append(float(CH_CHN0))
    CE_CEN0=C_Vapor[i]/C_Vapor_N[i]
    CE_CEN.append(float(CE_CEN0))    
    delta_virtemp0=virt_surftemp[i]-virt_airtemp[i]
    delta_virtemp.append(float(delta_virtemp0))
    delta_temp0=surftemp[i]-airtemp[i]
    delta_temp.append(float(delta_temp0))
    SenHeat_LatHeat_N0=SenHeat_N[i]/LatHeat_N[i]
    SenHeat_LatHeat_N.append(float(SenHeat_LatHeat_N0))
    SenHeat_LatHeat0=SenHeat[i]/LatHeat[i]
    SenHeat_LatHeat.append(float(SenHeat_LatHeat0))
    beta_betaN0=SenHeat_LatHeat0/(SenHeat_LatHeat_N0+0.001)
    beta_betaN.append(float(beta_betaN0))
    LatHeat_mmday0=LatHeat[i]/28.400
    LatHeat_mmday.append(float(LatHeat_mmday0))    
#st = surftemp - airtemp
#uStar_k = scipy.zeros(n)
#gradUxy = scipy.zeros(n)
#gradUyy = scipy.zeros(n)
#gradUzy = scipy.zeros(n)
#gradUTotal = scipy.zeros(n)
#U_x,U_y,U_z = meteolib.wind_comp(U,D)
#for i in range(0,n):
#    gradUxy[i] = C_Momentum[i]*airdensity_m[i]*math.pow(U_x[i],2)/(nu[i]*airdensity_m[i])
#    gradUyy[i] = C_Momentum[i]*airdensity_m[i]*math.pow(U_y[i],2)/(nu[i]*airdensity_m[i])
#    gradUzy[i] = C_Momentum[i]*airdensity_m[i]*math.pow(U_z[i],2)/(nu[i]*airdensity_m[i])
#    uStar_k[i] = math.sqrt(C_Momentum[i]*airdensity_m[i]*math.pow(U[i],2)/(1000.0))
#    if U_x[i]<>0:
#        gradUTotal[i] = U_x[i]/math.fabs(U_x[i])*math.sqrt(math.pow(gradUxy[i],2)+math.pow(gradUzy[i],2))
#    else:
#        gradUTotal[i] = math.sqrt(math.pow(gradUxy[i],2)+math.pow(gradUzy[i],2)) 
airdensity_m_avr = numpy.mean(airdensity_m)
airtemp_avr = numpy.mean(airtemp)
surftemp_avr = numpy.mean(surftemp)
Rs_avr = numpy.mean(Rs)
RH_avr = numpy.mean(RH)
U_wind_avr = numpy.mean(U)
D_avr = numpy.mean(D)
airpress_avr = numpy.mean(airpress)
z_momentum_avr = numpy.mean(z_momentum)
z_heat_avr = numpy.mean(z_heat)
z_vapor_avr = numpy.mean(z_vapor)
C_Momentum_avr = numpy.mean(C_Momentum)
C_Momentum_N_avr = numpy.mean(C_Momentum_N)
C_Heat_avr = numpy.mean(C_Heat)
C_Heat_N_avr = numpy.mean(C_Heat_N)
SenHeat_LatHeat_avr = numpy.mean(SenHeat_LatHeat)
SenHeat_N_avr=numpy.mean(SenHeat_N)
LatHeat_N_avr=numpy.mean(LatHeat_N)
SenHeat_avr=numpy.mean(SenHeat)
LatHeat_avr=numpy.mean(LatHeat)
unstable_count = float(len(filter(lambda x: x < 0, kesi)))/float(len(kesi))*100
stable_count = float(len(filter(lambda x: x >= 0, kesi)))/float(len(kesi))*100
freq_kesi_01 = float(len(filter(lambda x: x < -40.00, kesi)))
freq_kesi_02 = float(len(filter(lambda x: (x > -40.00) and (x < -30.00), kesi)))
freq_kesi_03 = float(len(filter(lambda x: (x > -30.00) and (x < -20.00), kesi)))
freq_kesi_04 = float(len(filter(lambda x: (x > -20.00) and (x < -10.00), kesi)))
freq_kesi_05 = float(len(filter(lambda x: (x > -10.00) and (x < -5.00), kesi)))
freq_kesi_06 = float(len(filter(lambda x: (x > -5.00) and (x < -2.00), kesi)))
freq_kesi_07 = float(len(filter(lambda x: (x > -2.00) and (x < -1.00), kesi)))
freq_kesi_08 = float(len(filter(lambda x: (x > -1.00) and (x <  0.00), kesi)))
freq_kesi_09 = float(len(filter(lambda x: (x > 0.00) and (x < 1.00), kesi)))
freq_kesi_10 = float(len(filter(lambda x: (x > 1.00) and (x < 2.00), kesi)))
freq_kesi_11 = float(len(filter(lambda x: (x > 2.00) and (x < 5.00), kesi)))
freq_kesi_12 = float(len(filter(lambda x: (x > 5.00) and (x < 10.00), kesi)))
freq_kesi_13 = float(len(filter(lambda x: (x > 10.00) and (x < 20.00), kesi)))
freq_kesi_14 = float(len(filter(lambda x: (x > 20.00) and (x < 30.00), kesi)))
freq_kesi_15 = float(len(filter(lambda x: (x > 30.00) and (x < 40.00), kesi)))
freq_kesi_16 = float(len(filter(lambda x: (x > 40.00) , kesi)))
freq_kesi =[freq_kesi_01,freq_kesi_02,freq_kesi_03,freq_kesi_04,freq_kesi_05,freq_kesi_06,\
            freq_kesi_07,freq_kesi_08,freq_kesi_09,freq_kesi_10,freq_kesi_11,freq_kesi_12,freq_kesi_13, \
            freq_kesi_14,freq_kesi_15,freq_kesi_16]
nn=scipy.size(freq_kesi)
for i in range(0, nn):
    freq_kesi_percent0=freq_kesi[i]/float(len(kesi))*100
    freq_kesi_percent.append(float(freq_kesi_percent0))
#-------------------------------------------------------------------
#-------------------------------------------------------------------------
#----------------------------------------------------------------------------------
df_analysed['C_p'] = cp
df_analysed['Lambda'] = Lambda
df_analysed['H_sim_stability'] = SenHeat
df_analysed['E_sim_stability'] = LatHeat
df_analysed['OB_L_stability'] = OB_L
df_analysed['H_N_sim_stability'] =SenHeat_N
df_analysed['E_N_sim_stability'] = LatHeat_N
df_analysed['u_Star_stability'] = u_Star
df_analysed['C_Momentum_stability'] = C_Momentum
df_analysed['C_Momentum_N_stability'] = C_Momentum_N
df_analysed['C_Heat_stability'] = C_Heat
df_analysed['C_Heat_N_stability'] = C_Heat_N
df_analysed['C_Vapor_stability'] = C_Vapor
df_analysed['C_Vapor_N_stability'] = C_Vapor_N
df_analysed['kesi_stability'] = kesi
df_analysed['z_momentum_stability'] = z_momentum
df_analysed['z_vapor_stability'] = z_vapor
df_analysed['z_heat_stability'] = z_heat
df_analysed['delta_e'] = df_analysed['es'] - df_analysed['ea']

df_analysed['H_HN_sim_stability'] = df_analysed['H_sim_stability'] / df_analysed['H_N_sim_stability']
df_analysed['E_EN_sim_stability'] = df_analysed['E_sim_stability'] / df_analysed['E_N_sim_stability'] 

# Generating dataframe for Daily Averaged values
df_analysed_dailyAve = df_analysed.resample('D', how='mean')

# Generating dataframe for Hourly Averaged values
df_analysed_hourlyAve = df_analysed.resample('H', how='mean')

# Generating dataframe for Monthly Averaged values
df_analysed_monthlyAve = df_analysed.resample('M', how='mean')

# Generating dataframe for Weekly Averaged values
df_analysed_weeklyAve = df_analysed.resample('W', how='mean')
#-------------------------------------------------------------------------------
n_size = scipy.size(df_analysed.index)
time_CFD =[]
stability_correct_mom = []
stability_correct_vapor = []
#q_CFD = []
#T_CFD = []
for i in range(0,n_size):
  time_CFD0 = df_analysed.index[i] - df_analysed.index[0]
  time_CFD0_sec =  time_CFD0.total_seconds()
  time_CFD.append(time_CFD0_sec)
  stability_correct_mom0 = totalLibs.stability_correct_mom(df_analysed.kesi_stability[i])
  stability_correct_vapor0 = totalLibs.stability_correct_vapor(df_analysed.kesi_stability[i])
  stability_correct_mom.append(stability_correct_mom0)
  stability_correct_vapor.append(stability_correct_vapor0)
 # Adding the new parameters to dataframe
df_analysed['time_CFD'] = time_CFD
df_analysed['stability_correct_mom'] = stability_correct_mom
df_analysed['stability_correct_vapor'] = stability_correct_vapor
# Define new data to plot from selected db
df_analysed_plot = df_analysed[(df_analysed.index >= start_Date) & (df_analysed.index <= end_Date_plot)]
# the path for saving the figures
save_add_lake = '../outputGraphs/' + reading_input.perfix 
#Writing text files for CFD Lake Model
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/z_mom.data',time_CFD,z_momentum)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/z_vap.data',time_CFD,z_vapor)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/z_heat.data',time_CFD,z_heat)
#writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/U2.data',time_CFD,df_Data_selPeriod['U2'])
#writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/U_y.data',time_CFD,df_Data_selPeriod['U2_y'])
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/ustar.data',time_CFD,u_Star)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/kesi.data',time_CFD,kesi)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/stability_mon.data',time_CFD,stability_correct_mom)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/stability_heat.data',time_CFD,stability_correct_vapor)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/stability_vapor.data',time_CFD,stability_correct_vapor)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/H.data',time_CFD,SenHeat)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/E.data',time_CFD,LatHeat)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/Xs.data',time_CFD,df_analysed['X_s'])
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/Xa.data',time_CFD,df_analysed['X_a'])
#-------------------------------------------------------------------------------
coeff = 3600
rt=scipy.size(dec_time)
r=675
rt=scipy.size(dec_time)
r_doy = r
first_day=315 # equal to Nivember 10, 2012
#dec_time_doy= dec_time/24 + first_day
y_label_size = 60 # the size of y label in graphs
x_label_size = 60 # the size of x label in graphs
z_label_size = 60 # the size of x label in graphs
legend_size = 60
fig_size=(18* 1.618, 18)
scatter_size_01 = 150
plot_linewidth_01=2.0
#------------------------------------------------------------------
plt.figure('z0_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
ax = plt.gca()
plt.ylabel(r'${\rm Roughness \, Length(z_{0}) [m] }$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.grid(True)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
ax.scatter(dec_time,z_momentum, label= r'$ z_{0m} $', s=scatter_size_01, marker='+',color='blue')
ax.scatter(dec_time,z_heat, label= r'$ z_{0h} $', s=scatter_size_01, marker='.',color='red')
#plt.show()
#ax.scatter(dec_time_doy[0:r],z_vapor[0:r], label= r'$ z_{0v} $', marker='*',color='red')
#pylab.legend()
#ax.set_yscale('log')
ax.set_ylim(0.00002,0.001)
#pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()

#plt.draw()
#plt.draw()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/z0_time.png',bbox_inches='tight')
print('z0 -----> Time created...\n')
#----------------------------------------------------------------
plt.figure('uStar_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm u_*~[m/s] }$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.scatter(dec_time,u_Star, label= r'$u_*$', s=scatter_size_01, marker='.',color='blue')
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/uStar_time.png',bbox_inches='tight')
#plt.show()
print('uStar -----> Time created...\n')
##-------------------------------------------------------------
plt.figure('Rs_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm Shortwave \, Radiation(R_s)[W \, m^{-2}] }$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
pylab.plot(dec_time,Rs, label= r'$R_s$', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/Rs_time.png',bbox_inches='tight')
#plt.show()
print('Rs -----> Time created...\n')
#-------------------------------------------------------------
plt.figure('T_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm Temperature~ [^\circ C]}$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.plot(dec_time,airtemp, label= r'$T_{\rm air}$', color='r', linewidth=plot_linewidth_01)
ax.plot(dec_time,surftemp, label= r'$T_{\rm surf}$', color='b', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/T_air_water_time.png',bbox_inches='tight')
#plt.show()
print('T air/surface -----> Time created...\n')
#-------------------------------------------------------------
plt.figure('E_H_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm Flux [W/m^2]}$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.plot(dec_time,LatHeat, label= r'$E$', color='b',linewidth=plot_linewidth_01)
ax.plot(dec_time,SenHeat, label= r'$H$', color='red', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/E_H_time.png',bbox_inches='tight')
#plt.show()
print('EandH -----> Time created...\n')
##-------------------------------------------------------------------
plt.figure('U_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm Wind \, Speed (U_2[m/s])}$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.plot(dec_time,U, label= r'$U_2$', color='b',linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/U_time.png',bbox_inches='tight')
#plt.show()
print('U -----> Time created...\n')
#---------------------------------------------------------------------------
fig = pylab.figure('EH_Tsurf_time',figsize=fig_size)
pylab.clf()
plt.cla()
ax1 = fig.add_subplot(111)
# recompute the ax.dataLim
ax1.relim()
# update ax.viewLim using the new dataLim
ax1.autoscale_view()
ax1.plot(dec_time, LatHeat, 'b-', label= r'$E$', linewidth=plot_linewidth_01)
ax1.plot(dec_time, SenHeat, 'r-', label= r'$H$', linewidth=plot_linewidth_01)
ax1.set_xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax1.set_ylabel(r'${\rm Heat \, Fluxes (H \, , E )[W/m^2]}$',size=y_label_size)
#for tl in ax1.get_yticklabels():
#    tl.set_color('b')
#pylab.legend()
ax1.legend(loc=1,prop={'size':legend_size})
ax2 = ax1.twinx()
# recompute the ax.dataLim
ax2.relim()
# update ax.viewLim using the new dataLim
ax2.autoscale_view()
ax2.plot(dec_time, surftemp,'g-',label= r'$T_{s}$', linewidth=plot_linewidth_01)
ax2.set_ylabel(r'${\rm Water \, Surface \, Temperature(T_s)[^\circ C]}$',size=y_label_size)
#for tl in ax2.get_yticklabels():
#    tl.set_color('r')
pylab.grid(True)
pylab.legend(loc=2)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
#plt.legend(loc='best',prop={'size':legend_size})
ax2.legend(loc=2, prop={'size':legend_size})
plt.gcf().autofmt_xdate()
#pylab.show()
pylab.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/EH_Tsurf_time.png',bbox_inches='tight')
print('E,H, T_surf------> time created...\n')
##-------------------------------------------------
pylab.figure('flux_rel_time',figsize=fig_size)
pylab.clf()
plt.cla()
# recompute the ax.dataLim
#plt.relim()
# update ax.viewLim using the new dataLim
#plt.autoscale_view()
pylab.ylabel(r'${\rm \frac{Flux}{Flux_{N}}}$',size=y_label_size)
pylab.xlabel(r'${\rm \zeta  }$',size=x_label_size)
pylab.scatter(kesi,LatHeat_Rel, label= r'$ E/E_{N} $', marker='+',s=scatter_size_01, color='blue')
pylab.scatter(kesi,SenHeat_Rel, label= r'$ H/H_{N} $', marker='.',s=scatter_size_01, color='red')
pylab.legend()
pylab.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
pylab.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/flux_rel_time.png',bbox_inches='tight')
#pylab.show()
print('Rel Flux -----> time created...\n')
#-------------------------------------------------------------------
pylab.figure('E_rel_time',figsize=fig_size)
pylab.clf()
plt.cla()
pylab.ylabel(r'${\rm \frac{E}{E_{N}}}$',size=y_label_size)
pylab.xlabel(r'${\rm \zeta  }$',size=x_label_size)
pylab.scatter(kesi,LatHeat_Rel, label= r'$ E/E_{N} $', marker='+',s=scatter_size_01, color='blue')
pylab.legend()
pylab.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
pylab.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/E_rel_time.png',bbox_inches='tight')
#pylab.show()
print('Rel E -----> time created...\n')
#-------------------------------------------------------------------
pylab.figure('H_rel_time',figsize=fig_size)
plt.cla()
pylab.clf()
pylab.ylabel(r'${\rm \frac{H}{H_{N}}}$',size=y_label_size)
pylab.xlabel(r'${\rm \zeta  }$',size=x_label_size)
pylab.scatter(kesi,SenHeat_Rel, label= r'$ H/H_{N} $', marker='.',s=scatter_size_01, color='red')
pylab.legend()
pylab.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
pylab.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/H_rel_time.png',bbox_inches='tight')
#pylab.show()
print('Rel H -----> time created...\n')
#--------------------------------------------------
plt.figure('z0m_U',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Momentum \, Roughness \, Length(z_{0m}) [m]} $',size=y_label_size)
plt.xlabel(r'${\rm Wind \, Speed (U_2) [m/s]}$',size=x_label_size)
ax.scatter(U,z_momentum, label= r'$ z_{0m} $', marker='.', s=scatter_size_01, color='blue')
#ax.set_yscale('log')
ax.set_ylim(1e-5,1e-3)
ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/z0m_U.png',bbox_inches='tight')
#plt.show()
print('z0m ---> U2 created...\n')
#--------------------------------------------------
plt.figure('z0v_U',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Vapour \, Roughness \, Length(z_{0v}) [m] }$',size=y_label_size)
plt.xlabel(r'${\rm Wind \, Speed (U_2) [m/s]}$',size=x_label_size)
ax.scatter(U,z_vapor, label= r'$ z_{0v} $', marker='.',s=scatter_size_01, color='blue')
#ax.set_yscale('log')
ax.set_ylim(2e-5,2e-3)
ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/z0v_U.png',bbox_inches='tight')
#plt.show()
print('z0v ----> U2 created...\n')
#--------------------------------------------------
plt.figure('z0h_U',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Heat \, Roughness \, Length(z_{0h}) [m] }$',size=y_label_size)
plt.xlabel(r'${\rm Wind \, Speed (U_2) [m/s]}$',size=x_label_size)
ax.scatter(U,z_heat, label= r'$ z_{0h} $', marker='+',s=scatter_size_01, color='blue')
#ax.set_yscale('log')
ax.set_ylim(2e-5,2e-3)
ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/z0h_U.png',bbox_inches='tight')
#plt.show()
print('z0h ----> U2 created...\n')
#--------------------------------------------------
plt.figure('uStar_U',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm u_*[m/s]}$',size=y_label_size)
plt.xlabel(r'${\rm U_2[m/s]}$',size=x_label_size)
ax.scatter(U, u_Star, label= r'$ u_* $',  marker='.',s=scatter_size_01, color='blue')
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/uStar_U.png',bbox_inches='tight')
#plt.show()
print('uStar ----> U created...\n') 
#-------------------------------------------------    
plt.figure('E_H_kesi',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Heat \, Flux(E,H) [Wm^{-2}]}$',size=y_label_size)
plt.xlabel(r'${\rm \zeta}$',size=x_label_size)
ax.scatter(kesi, SenHeat, label= r'$ H $',  marker='.',color='blue',s=scatter_size_01, linewidth=plot_linewidth_01)
ax.scatter(kesi, LatHeat, label= r'$ E $',  marker='.',color='red',s=scatter_size_01, linewidth=plot_linewidth_01)
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
#ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/E_H_kesi.png',bbox_inches='tight')
#plt.show()
print('E/H ----> kesi created...\n')     
#-------------------------------------------------    
plt.figure('E_kesi',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Heat \, Flux(E) [Wm^{-2}]}$',size=y_label_size)
plt.xlabel(r'${\rm \zeta}$',size=x_label_size)
ax.scatter(kesi, LatHeat, label= r'$ E $',  marker='.',color='red',s=scatter_size_01, linewidth=plot_linewidth_01)
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
#ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/E_kesi.png',bbox_inches='tight')
#plt.show()
print('E ----> kesi created...\n')     
#-------------------------------------------------    
plt.figure('H_kesi',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Heat \, Flux(H) [Wm^{-2}]}$',size=y_label_size)
plt.xlabel(r'${\rm \zeta}$',size=x_label_size)
ax.scatter(kesi, SenHeat, label= r'$ H $',  marker='.',color='blue',s=scatter_size_01, linewidth=plot_linewidth_01)
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
#ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/H_kesi.png',bbox_inches='tight')
#plt.show()
print('H ----> kesi created...\n')     
#------------------------------------------------------
plt.figure('E_U',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm E[mm \, day^{-1}]}$',size=y_label_size)
plt.xlabel(r'${\rm U_2[ms^{-1} ]}$',size=x_label_size)
ax.scatter(U, LatHeat_mmday, label= r'$ E $',  marker='.',s=scatter_size_01, color='blue',linewidth=plot_linewidth_01)
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
#ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/E_U.png',bbox_inches='tight')
#plt.show()
print('E ----> U created...\n')  
#------------------------------------------------
plt.figure('RH_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm Relative \, Humidity (RH)~[\%] }$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
pylab.plot(dec_time,RH, label= r'$RH$', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/RH_time.png',bbox_inches='tight')
#plt.show()
print('RH -----> Time created...\n')
#------------------------------------------------------    
plt.figure('HCal_HMeas_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm H ~[Wm^{-2}] }$', size=y_label_size)
plt.xlabel(r'${\rm Time []}$',size=x_label_size)
pylab.plot(dec_time,SenHeat, label= r'$H_{cal}$', linewidth=plot_linewidth_01, color='r')
pylab.plot(dec_time,H_m, label= r'$H_{meas}$', color='b', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/HCal_HMeas_time.png',bbox_inches='tight')
#plt.show()
print('H_cal_meas -----> Time created...\n')
#--------------------------------------------------------------------
plt.figure('uStarCal_uStarMeas_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm u^{*} ~[ms^{-1}] }$', size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
pylab.plot(dec_time,u_Star, label= r'$u^{*}_{Sim.}$', linewidth=plot_linewidth_01, color='r')
pylab.plot(dec_time,u_Star_m, label= r'$u^{*}_{Obs.}$', color='b', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/uStarCal_uStarMeas_time.png',bbox_inches='tight')
#plt.show()
print('uStar_cal_meas -----> Time created...\n')
#-------------------------------------------------------------------------------------------------
plt.figure('kesi_Cal_kesi_Meas_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm \zeta ~[-] }$', size=y_label_size)
plt.xlabel(r'${\rm Time []}$',size=x_label_size)
pylab.plot(dec_time,kesi, label= r'$\zeta _{Sim.}$', linewidth=plot_linewidth_01, color='r')
pylab.plot(dec_time,kessi_m, label= r'$\zeta _{Obs.}$', color='b', linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/kesi_Cal_kesi_Meas_time.png',bbox_inches='tight')
#plt.show()
print('kesi_cal_meas -----> Time created...\n')
#--------------------------------------------------------------------------------------------------------
print('*****  All Graphs were Plotted......***** \n')
#==========================================================

plt.figure('H_T',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm H[Wm^{-2}]}$',size=y_label_size)
plt.xlabel(r'${\rm (T_s-T_a)[K]}$',size=x_label_size)
ax.scatter((surftemp-airtemp), SenHeat, label= r'$ H $',  marker='.',color='red',s=scatter_size_01, linewidth=plot_linewidth_01)
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
#ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/H_T.png',bbox_inches='tight')
#plt.show()
print('H ----> (Ts-Ta) created...\n')  
#--------------------------------------------------------------------------------------------------------------------
#plt.figure('H_T',figsize=fig_size)
#plt.clf()
#plt.cla()
#plt.subplot(111)
#ax=plt.subplot(111)
#plt.ylabel(r'${\rm H[Wm^{-2}]}$',size=y_label_size)
#plt.xlabel(r'${\rm (T_s-T_a)[K]}$',size=x_label_size)
#ax.scatter((surftemp-airtemp), SenHeat, label= r'$ H $',  marker='.',color='red',s=scatter_size_01, linewidth=plot_linewidth_01)
##ax.set_yscale('log')
##ax.set_ylim(0.00145,0.00185)
##ax.set_ylim(0,6)
##ax.set_xlim(0,6)
#ax.grid(True)
#pylab.legend()
#pylab.gcf().autofmt_xdate()
#plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
#plt.legend(loc='best',prop={'size':legend_size})
#plt.gcf().autofmt_xdate()
#plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/H_T.png',bbox_inches='tight')
##plt.show()
#print('H ----> (Ts-Ta) created...\n')  
#========================================================
plt.figure('H_deltaT',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm H[Wm^{-2}]}$',size=y_label_size)
plt.xlabel(r'${\rm \Delta T [K]}$',size=x_label_size)
ax.scatter(delta_temp, SenHeat, label= r'$ H $',  marker='.',color='blue',s=scatter_size_01, linewidth=plot_linewidth_01)
#ax.set_yscale('log')
#ax.set_ylim(0.00145,0.00185)
#ax.set_ylim(0,6)
#ax.set_xlim(0,6)
ax.grid(True)
pylab.legend()
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/H_deltaT.png',bbox_inches='tight')
#plt.show()
print('H ----> deltaT created...\n')  
#------------------------------------------------
plt.figure('HCal_HMeas_evaluate',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm H_{cal} ~[Wm^{-2}] }$', size=y_label_size)
plt.xlabel(r'${\rm H_{meas} [Wm^{-2}]}$',size=x_label_size)
pylab.scatter(H_m, SenHeat, label= r'$H$', marker='.', s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.plot(H_m,H_m, 'b--' , linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/HCal_HMeas_time_evaluate.png',bbox_inches='tight')
#plt.show()
print('H_cal_meas -----> H_meas created...\n')
#-------------------------------------------------------------------------------------------------
plt.figure('HCal_HMeas',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm H_{cal} ~[Wm^{-2}] }$', size=y_label_size)
plt.xlabel(r'${\rm H_{meas} [Wm^{-2}]}$',size=x_label_size)
pylab.scatter(H_m, SenHeat, label= r'$H$', marker='.', s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/HCal_HMeas_time.png',bbox_inches='tight')
#plt.show()
print('H_cal_meas -----> H_meas created...\n')
#=======================================================
plt.figure('uStarCal_uStarMeas',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm u^{*}_{Sim.} ~[ms^{-1}] }$', size=y_label_size)
plt.xlabel(r'${\rm u^{*}_{Obs.} ~[ms^{-1}] }$',size=x_label_size)
pylab.scatter(u_Star_m,u_Star, label= r'$ u^{*}$', marker='.',s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/uStarCal_uStarMeas.png',bbox_inches='tight')
#plt.show()
print('uStar_cal -----> uStar_meas created...\n')
#----------------------------------------------
plt.figure('uStarCal_uStarMeas',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm u^{*}_{Sim.} ~[ms^{-1}] }$', size=y_label_size)
plt.xlabel(r'${\rm u^{*}_{Obs.} ~[ms^{-1}] }$',size=x_label_size)
pylab.scatter(u_Star_m,u_Star, label= r'$ u^{*}$', marker='.',s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/uStarCal_uStarMeas.png',bbox_inches='tight')
#plt.show()
print('uStar_cal -----> uStar_meas created...\n')
#==================================================
plt.figure('kesi_Cal_kesi_Meas',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm \zeta _{Sim.}}$', size=y_label_size)
plt.xlabel(r'${\rm \zeta _{Obs.}}$',size=x_label_size)
pylab.scatter(kessi_m,kesi, label= r'$\zeta$', marker='.', s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.legend()
#ax.set_yscale('log')
ax.set_ylim(-40,40)
ax.set_xlim(-40,40)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/kesi_Cal_kesi_Meas.png',bbox_inches='tight')
#plt.show()
print('kesi_cal -----> kesi_meas created...\n')
#--------------------------------------------------------------------------------------------------------
plt.figure('kesi_Cal_kesi_Meas',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm \zeta _{Sim.}}$', size=y_label_size)
plt.xlabel(r'${\rm \zeta _{Obs.}}$',size=x_label_size)
pylab.scatter(kessi_m,kesi, label= r'$\zeta$', marker='.', s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.legend()
#ax.set_yscale('log')
ax.set_ylim(-40,40)
ax.set_xlim(-40,40)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/kesi_Cal_kesi_Meas.png',bbox_inches='tight')
#plt.show()
print('kesi_cal -----> kesi_meas created...\n')
#===============================================
plt.figure('Hcal_to_Hobs_kesi_m',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
plt.ylabel(r'${\rm \frac{H_{Obs.}}{H_{Cal.}}}$', size=y_label_size)
plt.xlabel(r'${\rm \zeta }$',size=x_label_size)
pylab.scatter(df_analysed['kessi_m'], df_analysed['H_sim_stability']/df_analysed['senHeat_m'] , label= r'$H$', marker='.', s=scatter_size_01, linewidth=plot_linewidth_01, color='r')
pylab.legend()
#ax.set_yscale('log')
ax.set_ylim(-2,2)
ax.set_xlim(-40,10)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/Hcal_to_Hobs_kesi_m.png',bbox_inches='tight')
#plt.show()
print('H_cal_meas -----> H_meas created...\n')
#-------------------------------------------------------------------------------------------------
plt.figure('E_mmday_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm Daily \, Averagde \, Evaporation [mm/day]}$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.plot(df_analysed_dailyAve.index,df_analysed_dailyAve['E_sim_stability']/28.4, label= r'$E$', color='b',linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/E_mmday_time.png',bbox_inches='tight')
#plt.show()
print('E_mmday -----> Time created...\n')
##-------------------------------------------------------------------
plt.figure('X_time',figsize=fig_size)
plt.clf()
plt.cla()
plt.subplot(111)
ax=plt.subplot(111)
# recompute the ax.dataLim
ax.relim()
# update ax.viewLim using the new dataLim
ax.autoscale_view()
plt.ylabel(r'${\rm Water \, Vapour \, Mixing \, Ratio [kg(water)/kg(dry air)]}$',size=y_label_size)
plt.xlabel(r'${\rm Time [day]}$',size=x_label_size)
ax.plot(df_analysed.index,df_analysed['X_s'], label= r'$q_s$', color='b',linewidth=plot_linewidth_01)
ax.plot(df_analysed.index,df_analysed['X_a'], label= r'$q_a$', color='b',linewidth=plot_linewidth_01)
pylab.legend()
#ax.set_yscale('log')
#ax.set_ylim(0.00002,0.001)
#ax.set_xlim(315,r/(4*24) + 315)
ax.grid(True)
pylab.gcf().autofmt_xdate()
plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(reading_input.save_graphs +'/preparingABLModel_inputData/X_time.png',bbox_inches='tight')
plt.show()
print('X -----> Time created...\n')
##-------------------------------------------------------------------
print "E/E_N: \t" , df_analysed_dailyAve['E_EN_sim_stability'].mean(), "with kesi_mean: \t" , df_analysed_dailyAve['kesi_stability'].mean()
print "Max(E/E_N): \t" , df_analysed_dailyAve['E_EN_sim_stability'].max() , "with kesi: \t" , \
                         df_analysed_dailyAve.kesi_stability[df_analysed_dailyAve['E_EN_sim_stability'].idxmax()],"and with U2: \t" , \
                         df_analysed_dailyAve.U2[df_analysed_dailyAve['E_EN_sim_stability'].idxmax()]
print "Min(E/E_N): \t" ,df_analysed_dailyAve['E_EN_sim_stability'].min(), \
      "with kesi_mean: \t" ,df_analysed_dailyAve.kesi_stability[df_analysed_dailyAve['E_EN_sim_stability'].idxmin()]
print "E(average)=" , df_analysed_dailyAve['E_sim_stability'].mean(), float(df_analysed_dailyAve['E_sim_stability'].mean()/28.4)
print "E_N(average)=" , df_analysed_dailyAve['E_N_sim_stability'].mean(),float(df_analysed_dailyAve['E_N_sim_stability'].mean()/28.4)
print "Average of Air Density: \t" , airdensity_m_avr
print "Average of Wind velocity: \t" , U_wind_avr
print "Average of Air Temperature: \t" ,  airtemp_avr
print "Average of Water Surface temp.: \t" ,  surftemp_avr
print "Average of Short Radiation: \t" ,  Rs_avr
print "Average of RH: \t" ,  RH_avr
print "Average of D: \t" ,  D_avr
print "Average of Patm: \t" ,  airpress_avr
print "Average of z_momentum: \t" ,  z_momentum_avr
print "Average of z_heat: \t" ,  z_heat_avr
print "Average of z_vapor: \t" ,  z_vapor_avr
print "Average of C_Momentum: \t" ,  C_Momentum_avr
print "Average of C_Momentum_N: \t" ,  C_Momentum_N_avr
print "Average of C_Momentum/C_Momentum_N: \t" ,  float(C_Momentum_avr/C_Momentum_N_avr*100)
print "Average of C_Heat: \t" ,  C_Heat_avr
print "Average of C_Heat_N: \t" ,  C_Heat_N_avr
print "Average of C_Heat/C_Heat_N: \t" ,  float(C_Heat_avr/C_Heat_N_avr*100)
print "Average of Sensible Heat[W m-2]: \t" , SenHeat_avr
print "Average of Latent Heat[w m-2]: \t" , LatHeat_avr
print "Average of Neutral Sensible Heat[w m-2]: \t" , SenHeat_N_avr
print "Average of Neutral Latent Heat[wm-2]: \t" , LatHeat_N_avr
print "Average of Evaporation[mm/day]: \t" , float(LatHeat_avr/28.4)
print "Average of Neutral Evaporation[mm/day]: \t" , float(LatHeat_N_avr/28.4)
print "Average of Evaporation / Neutral Evaporation[-]: \t" , float(LatHeat_avr/LatHeat_N_avr)
print "Average of beta: \t" ,  float(SenHeat_LatHeat_avr)
print "Stable Condition[%]: \t" ,  stable_count
print "Unstable Condition[%]: \t" ,  unstable_count
print "Average of freq_kesi: \t" ,  freq_kesi_01/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_02/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_03/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_04/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_05/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_06/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_07/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_08/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_09/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_10/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_11/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_12/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_13/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_14/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_15/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  freq_kesi_16/float(len(kesi))*100
print "Average of freq_kesi: \t" ,  numpy.sum(freq_kesi)/float(len(kesi))*100
print df_analysed.describe()
