# -*- coding: utf-8 -*-

# Linear Regression from data reading from text file
"""
Created on Thu Sep 27 15:23:16 2012
@author: aliabbasi
"""
import pandas as pd  
import glob  
import matplotlib.pyplot as plt
#import pandas.tools.rplot as rplot
import pylab 
import matplotlib 
#import meteolib
import os    # operating system routines
import scipy # Scientific Python functions
import pylab # Plotting functions
import math
import numpy as np
import matplotlib.dates as mdates
from matplotlib import rc, rcParams
from datetime import datetime
import meteolib
import pdb
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.tools 
from statsmodels.tools.eval_measures import (
    maxabs, meanabs, medianabs, medianbias, mse, rmse, stde, vare,
    aic, aic_sigma, aicc, aicc_sigma, bias, bic, bic_sigma,
    hqic, hqic_sigma, iqr)
from sklearn.metrics import accuracy_score
import datetime
import reading_input
#from pandaslib import Timestamp
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
        filelist.append(str('%8.2f' % float(y1[i])))
        filelist.append(')')
        #filelist.append('\t')
        #filelist.append(str('%5.2f' % float(y2[i])))
        filelist.append('\n')
        outputfile.writelines(filelist)
    outputfile.write (')\n')    
    outputfile.close()
    
    return()
#-----------------------------------------------------------------------------------
#=====================================================================================
##### plot appearance setup
linewidth = 1.5
plt.rc("axes", linewidth=linewidth)
plt.rc("font", family="")
plt.rc("axes", titlesize="small", labelsize="small")
plt.rc("xtick", labelsize="x-small")
plt.rc("xtick.major", width=linewidth/2)
plt.rc("ytick", labelsize="x-small")
plt.rc("ytick.major", width=linewidth/2)
plt.rc("legend", fontsize="small")

#### save plot as PGF source (no setup needed)

#---------------
### setup matplotlib to save PDF with the PGF backend (tricky in Julia)
#backend_pgf = pyimport("matplotlib.backends.backend_pgf")
#backend_bases = pyimport("matplotlib.backend_bases")
#backend_bases[:register_backend]("pdf", backend_pgf[:FigureCanvasPgf])

### setup PGF-PDF backend output appearance
#plt.rc("pgf", texsystem="pdflatex",
#              preamble=L"""\usepackage[utf8x]{inputenc}
#                           \usepackage[T1]{fontenc}
#                           \usepackage{lmodern}""")

#### save directly as PDF (there is NO need to save as PFG first)
#plt.savefig("test.pdf", transparent=true)

#====================================================================================
pylab.rc('text', usetex=True)
# Change default settings for figures
newdefaults = {'fontname':    'Arial',  # Use Arial font
               'backend':       'png',  # Save figure as EPS file   
               'axes.labelsize':   30,  # Axis label size in points
               'text.fontsize':    30,  # Text size in points
               'legend.fontsize':  35,  # Legend label size in points
               'xtick.labelsize':  30,  # x-tick label size in points
               'ytick.labelsize':  30,  # y-tick label size in points
               'lines.markersize': 30,  # markersize, in points
               'lines.linewidth':   1.5 # line width in points
               }
pylab.rcParams.update(newdefaults)
# =================================================================================
#pd.options.display.mpl_style = 'default'
# Defining the range of data to prepare the inout file for CFD Lake Model
#start_Date = '11/23/2012 23:59:00'
#end_Date = '11/30/2012 23:59:53'
#end_Date_plot = '11/25/2012 23:59:53'
start_Date = '12/1/2012 23:59:00'# '12/15/2012 23:59:54'
end_Date = '12/3/2012 23:59:53' # '12/20/2012 23:59:53'
end_Date_plot = '12/3/2012 23:59:53' #  '12/20/2012 23:59:53'
# generating a new DataFrame according to the desired range from start to end date
df_Data_selPeriod = reading_input.df_totalData[(reading_input.df_totalData.index >= start_Date) & (reading_input.df_totalData.index <= end_Date)]

n_size = scipy.size(df_Data_selPeriod.index)

time_CFD =[]
U_x =[]
U_y =[]

for i in range(0,n_size):
  time_CFD0 = df_Data_selPeriod.index[i] - df_Data_selPeriod.index[0]
  time_CFD0_sec =  time_CFD0.total_seconds()
  time_CFD.append(time_CFD0_sec)
  #U_x0 , U_y0  = meteolib.wind_comp(df_Data_selPeriod.U2[i],df_Data_selPeriod.windDir[i]) 
  #U_y0 = U[i] *math.sin(math.radians(D[i])+math.pi)
  #U_x.append(float(U_x0))
  #U_y.append(float(U_y0))
  #print time_CFD[i],U[i],U_x[i],U_y[i]

# Adding the new parameters to dataframe
#df_Data_selPeriod['U2_x'] = U_x
#df_Data_selPeriod['U2_y'] = U_y
df_Data_selPeriod['time_CFD'] = time_CFD

# Define new data to plot from selected db
df_Data_selPeriod_plot = df_Data_selPeriod[(df_Data_selPeriod.index >= start_Date) & (df_Data_selPeriod.index <= end_Date_plot)]
# the path for saving the figures
save_add_lake = '../outputGraphs/' + reading_input.perfix 

#Writing text files for CFD Lake Model
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/Tair.data',time_CFD,df_Data_selPeriod['air_Temp']+273.15)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/RH.data',time_CFD,df_Data_selPeriod['RH'])
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/sRadiation.data',time_CFD,df_Data_selPeriod['Rs'])
#writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/U2.data',time_CFD,df_Data_selPeriod['U2'])
#writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/U_y.data',time_CFD,df_Data_selPeriod['U2_y'])
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/U.data',time_CFD,df_Data_selPeriod['U2'])
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/Ts.data',time_CFD,df_Data_selPeriod['surf_Temp']+273.15)
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/H.data',time_CFD,df_Data_selPeriod['senHeat_m'])
writeData_textFile_ABLModel(save_add_lake + '/preparingABLModel_inputData/U_Dir.data',time_CFD,df_Data_selPeriod['windDir'])
#***********************************************************************************
# Doing Statistics Analysis of Data(selected dataframe)
print "parameter \t" , "Count \t" ,"mean \t" ,"mad \t" , "max \t" ,"min \t" ,"std\t" ,"var " 
print "Air Temp. " ,'%8.0f' %df_Data_selPeriod.air_Temp.count(),'%8.2f' %df_Data_selPeriod.air_Temp.mean(),'%7.2f' %df_Data_selPeriod.air_Temp.mad(),'%8.2f' %df_Data_selPeriod.air_Temp.max(), '%7.2f' %df_Data_selPeriod.air_Temp.min(),'%6.2f' %df_Data_selPeriod.air_Temp.std(), '%8.2f' %df_Data_selPeriod.air_Temp.var()

print "WS Temp. " ,'%8.0f' %df_Data_selPeriod.surf_Temp.count(),'%8.2f' %df_Data_selPeriod.surf_Temp.mean(),'%7.2f' %df_Data_selPeriod.surf_Temp.mad(),'%8.2f' %df_Data_selPeriod.surf_Temp.max(), '%7.2f' %df_Data_selPeriod.surf_Temp.min(),'%6.2f' %df_Data_selPeriod.surf_Temp.std(), '%8.2f' %df_Data_selPeriod.surf_Temp.var()

print "RH " ,'%8.0f' %df_Data_selPeriod.RH.count(),'%8.2f' %df_Data_selPeriod.RH.mean(),'%7.2f' %df_Data_selPeriod.RH.mad(),'%8.2f' %df_Data_selPeriod.RH.max(), '%7.2f' %df_Data_selPeriod.RH.min(),'%6.2f' %df_Data_selPeriod.RH.std(), '%8.2f' %df_Data_selPeriod.RH.var()

print "U " ,'%8.0f' %df_Data_selPeriod.U2.count(),'%8.2f' %df_Data_selPeriod.U2.mean(),'%7.2f' %df_Data_selPeriod.U2.mad(),'%8.2f' %df_Data_selPeriod.U2.max(), '%7.2f' %df_Data_selPeriod.U2.min(),'%6.2f' %df_Data_selPeriod.U2.std(), '%8.2f' %df_Data_selPeriod.U2.var()

print "Rs " ,'%8.0f' %df_Data_selPeriod.Rs.count(),'%8.2f' %df_Data_selPeriod.Rs.mean(),'%7.2f' %df_Data_selPeriod.Rs.mad(),'%8.2f' %df_Data_selPeriod.Rs.max(), '%7.2f' %df_Data_selPeriod.Rs.min(),'%6.2f' %df_Data_selPeriod.Rs.std(), '%8.2f' %df_Data_selPeriod.Rs.var()

print "U_Dir " ,'%8.0f' %df_Data_selPeriod.windDir.count(),'%8.2f' %df_Data_selPeriod.windDir.mean(),'%7.2f' %df_Data_selPeriod.windDir.mad(),'%8.2f' %df_Data_selPeriod.windDir.max(), '%7.2f' %df_Data_selPeriod.windDir.min(),'%6.2f' %df_Data_selPeriod.windDir.std(), '%8.2f' %df_Data_selPeriod.windDir.var()
#--------------------------------------------------------------------------------------
#Plotting parameters
x_lable_size = 50
y_lable_size = 50
legend_size = 50
titel_size = 50
up = scipy.size(reading_input.data_size)
low = 0
figure_size=(10* 1.618, 10)
#===================================================================================
# Plotting the graphs to put in the report
#-----------------------------------------------------------------------------
#Plotting parameters
x_lable_size = 30
y_lable_size = 30
legend_size = 30
titel_size = 30
up = scipy.size(reading_input.data_size)
low = 0
figure_size=(10* 1.618, 10)
#----------------------------------------------------------------
save_figin= '/media/localadmin/UserData/SyncronizingData/Dropbox/Ph.D_WorkFlows/Python_modelling/finalPythonCodes/outputGraphs/Binaba 23Nov-21Dec Reprocessed/'
#----------------------------------------------------------------------
pylab.figure('U2_time_plot',figsize=figure_size)
pylab.clf()
pylab.ylabel(r'${\rm U_2[m/s] }$',size=y_lable_size)
pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
plt.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['U2'], color='b', linewidth=1.5,label= r'$ wind \, speed $')
pylab.gcf().autofmt_xdate()
plt.grid(color='black', alpha=0.4, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(save_figin + '/preparingABLModel_inputData/U2_time_plot.svg',bbox_inches='tight')
#plt.show()
#----------------------------------------------------------------
pylab.figure('RH_time_plot',figsize=figure_size)
pylab.clf()
pylab.ylabel(r'${\rm RH[\%] }$',size=y_lable_size)
pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
plt.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['RH'], color='b', linewidth=1.5,label= r'$ Relative \, Humidity $')
pylab.gcf().autofmt_xdate()
plt.grid(color='black', alpha=0.4, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(save_figin + '/preparingABLModel_inputData/RH_time_plot.svg',bbox_inches='tight')
#plt.show()
#---------------------------------------------------------------- 
fig = pylab.figure('air_water_Temp_time_plot',figsize=figure_size)
pylab.clf()
ax1 = fig.add_subplot(111)
ax1.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['air_Temp'], color='b', linewidth=1.5,label= r'$ T_{air} $')
ax1.set_xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
ax1.set_ylabel(r'${\rm Air \, Temperature[C] }$',size=y_lable_size)
ax1.legend(loc=1,prop={'size':legend_size})
ax2 = ax1.twinx()
ax2.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['surf_Temp'], color='r', linewidth=1.5,label= r'$ T_{water\,surface} $')
ax2.set_ylabel(r'${\rm Water\, Surface \, Temperature[C] }$',size=y_lable_size)
pylab.grid(True)
ax2.legend(loc=2, prop={'size':legend_size})
pylab.gcf().autofmt_xdate()
ax1.grid(color='black', alpha=0.4, linestyle='dashed', linewidth=0.3)
#plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(save_figin + '/preparingABLModel_inputData/air_water_Temp_time_plot.svg',bbox_inches='tight')
#plt.show() 
#----------------------------------------------------------------  
pylab.figure('Tairwater_time_plot',figsize=figure_size)
pylab.clf()
pylab.ylabel(r'${\rm Temperature[C] }$',size=y_lable_size)
pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
plt.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['air_Temp'], color='b', linewidth=1.5,label= r'$ T_{air} $')
plt.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['surf_Temp'], color='r', linewidth=1.5,label= r'$ T_{water\,surface} $')
pylab.gcf().autofmt_xdate()
plt.grid(color='black', alpha=0.4, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(save_figin + '/preparingABLModel_inputData/Tairwater_time_plot.svg',bbox_inches='tight')
#plt.show() 
#----------------------------------------------------------------      
pylab.figure('Rs_time_plot',figsize=figure_size)
pylab.clf()
pylab.ylabel(r'${\rm Short \, Wave \, Radiation[W\, m^{-2}] }$',size=y_lable_size)
pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
plt.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['Rs'], color='b', linewidth=1.5, label= r'$ Short \, Wave \, Radiation $')
pylab.gcf().autofmt_xdate()
plt.grid(color='black', alpha=0.4, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(save_figin + '/preparingABLModel_inputData/Rs_time_plot.svg',bbox_inches='tight')
#plt.show()
#----------------------------------------------------------------   
pylab.figure('U_Dir_time_plot',figsize=figure_size)
pylab.clf()
pylab.ylabel(r'${\rm Wind \, Direction [Degree] }$',size=y_lable_size)
pylab.xlabel(r'${\rm Time[]}$',size=x_lable_size)
plt.plot(df_Data_selPeriod_plot.index, df_Data_selPeriod_plot['windDir'], color='b', linewidth=1.5, label= r'$ Wind \, Direction $')
pylab.gcf().autofmt_xdate()
plt.grid(color='black', alpha=0.4, linestyle='dashed', linewidth=0.3)
plt.legend(loc='best',prop={'size':legend_size})
plt.gcf().autofmt_xdate()
plt.savefig(save_figin + '/preparingABLModel_inputData/U_Dir_time_plot.svg',bbox_inches='tight')
plt.show()
#----------------------------------------------------------------   
    
