# Linear Regression from data reading from text file
"""
Created on Monday June 15 17:00:00 2015
@author: Ali Abbasi [aliabbasi.civileng@gmail.com]
"""
#from __future__ import print_function
import pdb
import os    # operating system routines
import scipy # Scientific Python functions
from scipy import stats
import pylab # Plotting functions
import math
#import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import meteolib
import reading_input

#----------------------------------------------------------------------
pylab.rc('text', usetex=True)
# Change default settings for figures
newdefaults = {'font.family' : 'serif',
               'font.serif': 'Times New Roman',
               #'fontname':    'Times New Roman',  # Use Arial font
               'backend':       'tif',  # Save figure as EPS file   
               'axes.labelsize':   60,  # Axis label size in points
               'text.fontsize':    60,  # Text size in points
               'legend.fontsize':  50,  # Legend label size in points
               'xtick.labelsize':  60,  # x-tick label size in points
               'ytick.labelsize':  60,  # y-tick label size in points
               'lines.markersize': 60,  # markersize, in points
               'lines.linewidth':   2.5 # line width in points
               }
pylab.rcParams.update(newdefaults)
#----------------------------------------------------------
#Plotting parameters
x_lable_size = 60
y_lable_size = 60
legend_size = 55
titel_size = 70
figure_size=(20* 1.618, 20)
x_origin = 500.00
z_origin = 4.04
#------------------------------------------------------------------------------------------------------------------------------------------
def reading_sample(case_name,point_number):
    path_sample = '/media/localadmin/Seagate_4T/' + case_name + '/postProcessing/sets/'
    t0 = '0/'
    t1 = '3660/'
    t2 = '7260/'
    t3 = '10860/'
    t4 = '14460/'
    t5 = '18060/'
    t6 = '21660/'
    t7 = '25260/'
    t8 = '28860/'
    t9 = '32460/'
    t10 = '36060/'
    t11 = '39660/'
    t12 = '43260/'
    t13 = '46860/'
    t14 = '50460/'
    t15 = '54060/'
    t16 = '57660/'
    t17 = '61460/'
    t18 = '64760/'
    t19 = '68660/'
    t20 = '71960/'
    t21 = '86360/'
    t22 = '100760/'
    t23 = '102260/'
   
    # Readong one sample line
    df_samp_t0 = pd.read_csv(path_sample + t0 + point_number + '_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t0','Ux_t0','Uy_t0','Uz_t0','alphat_t0','epsilon_t0','k_t0','nut_t0','p_t0','p_rgh_t0', 'q_t0'] , header=0)
    df_samp_t1 = pd.read_csv(path_sample + t1  + point_number + '_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t1','Ux_t1','Uy_t1','Uz_t1','alphat_t1','epsilon_t1','k_t1','nut_t1','p_t1','p_rgh_t1', 'q_t1'] , header=0)
    df_samp_t2 = pd.read_csv(path_sample + t2  + point_number + '_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t2','Ux_t2','Uy_t2','Uz_t2','alphat_t2','epsilon_t2','k_t2','nut_t2','p_t2','p_rgh_t2', 'q_t2'] , header=0)
    df_samp_t3 = pd.read_csv(path_sample + t3  + point_number+ '_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t3','Ux_t3','Uy_t3','Uz_t3','alphat_t3','epsilon_t3','k_t3','nut_t3','p_t3','p_rgh_t3', 'q_t3'] , header=0)
    df_samp_t4 = pd.read_csv(path_sample + t4  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t4','Ux_t4','Uy_t4','Uz_t4','alphat_t4','epsilon_t4','k_t4','nut_t4','p_t4','p_rgh_t4', 'q_t4'] , header=0)
    df_samp_t5 = pd.read_csv(path_sample + t5  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t5','Ux_t5','Uy_t5','Uz_t5','alphat_t5','epsilon_t5','k_t5','nut_t5','p_t5','p_rgh_t5', 'q_t5'] , header=0)
    df_samp_t6 = pd.read_csv(path_sample + t6  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t6','Ux_t6','Uy_t6','Uz_t6','alphat_t6','epsilon_t6','k_t6','nut_t6','p_t6','p_rgh_t6', 'q_t6'] , header=0)
    df_samp_t7 = pd.read_csv(path_sample + t7  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t7','Ux_t7','Uy_t7','Uz_t7','alphat_t7','epsilon_t7','k_t7','nut_t7','p_t7','p_rgh_t7', 'q_t7'] , header=0)
    df_samp_t8 = pd.read_csv(path_sample + t8  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t8','Ux_t8','Uy_t8','Uz_t8','alphat_t8','epsilon_t8','k_t8','nut_t8','p_t8','p_rgh_t8', 'q_t8'] , header=0)
    df_samp_t9 = pd.read_csv(path_sample + t9  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t9','Ux_t9','Uy_t9','Uz_t9','alphat_t9','epsilon_t9','k_t9','nut_t9','p_t9','p_rgh_t9', 'q_t9'] , header=0)
    df_samp_t10 = pd.read_csv(path_sample + t10  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t10','Ux_t10','Uy_t10','Uz_t10','alphat_t10','epsilon_t10','k_t10','nut_t10','p_t10','p_rgh_t10', 'q_t10'] , header=0)
    df_samp_t11 = pd.read_csv(path_sample + t11  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t11','Ux_t11','Uy_t11','Uz_t11','alphat_t11','epsilon_t11','k_t11','nut_t11','p_t11','p_rgh_t11', 'q_t11'] , header=0)
    df_samp_t12 = pd.read_csv(path_sample + t12 + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t12','Ux_t12','Uy_t12','Uz_t12','alphat_t12','epsilon_t12','k_t12','nut_t12','p_t12','p_rgh_t12', 'q_t12'] , header=0)
    df_samp_t13 = pd.read_csv(path_sample + t13  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t13','Ux_t13','Uy_t13','Uz_t13','alphat_t13','epsilon_t13','k_t13','nut_t13','p_t13','p_rgh_t13', 'q_t13'] , header=0)
    df_samp_t14 = pd.read_csv(path_sample + t14  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t14','Ux_t14','Uy_t14','Uz_t14','alphat_t14','epsilon_t14','k_t14','nut_t14','p_t14','p_rgh_t14', 'q_t14'] , header=0)
    df_samp_t15 = pd.read_csv(path_sample + t15  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t15','Ux_t15','Uy_t15','Uz_t15','alphat_t15','epsilon_t15','k_t15','nut_t15','p_t15','p_rgh_t15', 'q_t15'] , header=0)
    df_samp_t16 = pd.read_csv(path_sample + t16  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t16','Ux_t16','Uy_t16','Uz_t16','alphat_t16','epsilon_t16','k_t16','nut_t16','p_t16','p_rgh_t16', 'q_t16'] , header=0)
    df_samp_t17 = pd.read_csv(path_sample + t17  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t17','Ux_t17','Uy_t17','Uz_t17','alphat_t17','epsilon_t17','k_t17','nut_t17','p_t17','p_rgh_t17', 'q_t17'] , header=0)
    df_samp_t18 = pd.read_csv(path_sample + t18  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t18','Ux_t18','Uy_t18','Uz_t18','alphat_t18','epsilon_t18','k_t18','nut_t18','p_t18','p_rgh_t18', 'q_t18'] , header=0)
    df_samp_t19 = pd.read_csv(path_sample + t19  + point_number+ '_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t19','Ux_t19','Uy_t19','Uz_t19','alphat_t19','epsilon_t19','k_t19','nut_t19','p_t19','p_rgh_t19', 'q_t19'] , header=0)
    df_samp_t20 = pd.read_csv(path_sample + t20  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t20','Ux_t20','Uy_t20','Uz_t20','alphat_t20','epsilon_t20','k_t20','nut_t20','p_t20','p_rgh_t20', 'q_t20'] , header=0)
    df_samp_t21 = pd.read_csv(path_sample + t21  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t21','Ux_t21','Uy_t21','Uz_t21','alphat_t21','epsilon_t21','k_t21','nut_t21','p_t21','p_rgh_t21', 'q_t21'] , header=0)
    df_samp_t22 = pd.read_csv(path_sample + t22  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t22','Ux_t22','Uy_t22','Uz_t22','alphat_t22','epsilon_t22','k_t22','nut_t22','p_t22','p_rgh_t22', 'q_t22'] , header=0)
    df_samp_t23 = pd.read_csv(path_sample + t23  + point_number +'_T_Ux_Uy_Uz_alphat_epsilon_k_nut_p_p_rgh_q.csv',  index_col=0,sep=",",skiprows=0, \
                             names =['z','T_t23','Ux_t23','Uy_t23','Uz_t23','alphat_t23','epsilon_t23','k_t23','nut_t23','p_t23','p_rgh_t23', 'q_t23'] , header=0)
#    df_samp_U_t0 = pd.read_csv(path_sample + t0 + point_number + '_U.csv',  index_col=0,sep=",",skiprows=0, names =['z','Ux_t0','Uy_t0','Uz_t0'] , header=0)
#    df_samp_U_t1 = pd.read_csv(path_sample + t1 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t1','Uy_t1','Uz_t1'] , header=0)
#    df_samp_U_t2 = pd.read_csv(path_sample + t2 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t2','Uy_t2','Uz_t2'] , header=0)
#    df_samp_U_t3 = pd.read_csv(path_sample + t3 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t3','Uy_t3','Uz_t3'] , header=0)
#    df_samp_U_t4 = pd.read_csv(path_sample + t4 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t4','Uy_t4','Uz_t4'] , header=0)
#    df_samp_U_t5 = pd.read_csv(path_sample + t5 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t5','Uy_t5','Uz_t5'] , header=0)
#    df_samp_U_t6 = pd.read_csv(path_sample + t6 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t6','Uy_t6','Uz_t6'] , header=0)
#    df_samp_U_t7 = pd.read_csv(path_sample + t7 + point_number + '_U.csv',index_col=0,  sep=",",skiprows=0, names =['z','Ux_t7','Uy_t7','Uz_t7'] , header=0)
#    df_samp_U_t8 = pd.read_csv(path_sample + t8 + point_number + '_U.csv',index_col=0,  sep=",",skiprows=0, names =['z','Ux_t8','Uy_t8','Uz_t8'] , header=0)
#    df_samp_U_t9 = pd.read_csv(path_sample + t9 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t9','Uy_t9','Uz_t9'] , header=0)
#    df_samp_U_t10 = pd.read_csv(path_sample + t10 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t10','Uy_t10','Uz_t10'] , header=0)
#    df_samp_U_t11 = pd.read_csv(path_sample + t11 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t11','Uy_t11','Uz_t11'] , header=0)
#    df_samp_U_t12 = pd.read_csv(path_sample + t12 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t12','Uy_t12','Uz_t12'] , header=0)
#    df_samp_U_t13 = pd.read_csv(path_sample + t13 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t13','Uy_t13','Uz_t13'] , header=0)
#    df_samp_U_t14 = pd.read_csv(path_sample + t14 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t14','Uy_t14','Uz_t14'] , header=0)
#    df_samp_U_t15 = pd.read_csv(path_sample + t15 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t15','Uy_t15','Uz_t15'] , header=0)
#    df_samp_U_t16 = pd.read_csv(path_sample + t16 + point_number + '_U.csv',index_col=0,  sep=",",skiprows=0, names =['z','Ux_t16','Uy_t16','Uz_t16'] , header=0)
#    df_samp_U_t17 = pd.read_csv(path_sample + t17 + point_number + '_U.csv',index_col=0,  sep=",",skiprows=0, names =['z','Ux_t17','Uy_t17','Uz_t17'] , header=0)
#    df_samp_U_t18 = pd.read_csv(path_sample + t18 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t18','Uy_t18','Uz_t18'] , header=0)
#    df_samp_U_t19 = pd.read_csv(path_sample + t19 + point_number + '_U.csv', index_col=0, sep=",",skiprows=0, names =['z','Ux_t19','Uy_t19','Uz_t19'] , header=0)
#    df_samp_U_t20 = pd.read_csv(path_sample + t20 + point_number + '_U.csv',index_col=0,  sep=",",skiprows=0, names =['z','Ux_t20','Uy_t20','Uz_t20'] , header=0)
    
#    df_samp = pd.concat([df_samp_t0,df_samp_t1,df_samp_t2,df_samp_t3,df_samp_t4,df_samp_t5,df_samp_t6,df_samp_t7,df_samp_t8, \
#                            df_samp_t9,df_samp_t10,df_samp_t11,df_samp_t12,df_samp_t13,df_samp_t14,df_samp_t15,df_samp_t16,df_samp_t17, \
#                             df_samp_t18,df_samp_t19,df_samp_t20, \
#                             df_samp_U_t0,df_samp_U_t1,df_samp_U_t2,df_samp_U_t3,df_samp_U_t4,df_samp_U_t5,df_samp_U_t6,df_samp_U_t7,df_samp_U_t8, \
#                            df_samp_U_t9,df_samp_U_t10,df_samp_U_t11,df_samp_U_t12,df_samp_U_t13,df_samp_U_t14,df_samp_U_t15,df_samp_U_t16,df_samp_U_t17, \
#                             df_samp_U_t18,df_samp_U_t19,df_samp_U_t20], axis=1)    
    df_samp = pd.concat([df_samp_t0,df_samp_t1,df_samp_t2,df_samp_t3,df_samp_t4,df_samp_t5,df_samp_t6,df_samp_t7,df_samp_t8, \
                            df_samp_t9,df_samp_t10,df_samp_t11,df_samp_t12,df_samp_t13,df_samp_t14,df_samp_t15,df_samp_t16,df_samp_t17, \
                             df_samp_t18,df_samp_t19,df_samp_t20,df_samp_t21,df_samp_t22,df_samp_t23], axis=1)    
    print df_samp.describe()    
    return df_samp
#------------------------------------------------------------
def reading_probe(case_name):
    path_probe = '/media/localadmin/Seagate_4T/'  + case_name +'/postProcessing/probes/'
    df_T_probe = pd.read_csv(path_probe + '0/T.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'T01': np.float64  ,'T02': np.float64 ,'T03': np.float64 ,'T04': np.float64 ,'T05': np.float64 , \
             'T06': np.float64 ,'T07': np.float64 ,'T08': np.float64 ,'T09': np.float64  }, \
              names =['time_plot','T01' ,'T02','T03','T04','T05','T06','T07','T08','T09'], header =0)
    df_k_probe = pd.read_csv(path_probe + '0/k.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'k01': np.float64  ,'k02': np.float64 ,'k03': np.float64 ,'k04': np.float64 ,'k05': np.float64 , \
             'k06': np.float64 ,'k07': np.float64 ,'k08': np.float64 ,'k09': np.float64 }, \
              names =['time_plot','k01' ,'k02','k03','k04','k05','k06','k07','k08','k09'], header =0)              
    df_alphat_probe = pd.read_csv(path_probe + '0/alphat.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'alphat01': np.float64  ,'alphat02': np.float64 ,'alphat03': np.float64 ,'alphat04': np.float64 ,'alphat05': np.float64 , \
             'alphat06': np.float64 ,'alphat07': np.float64 ,'alphat08': np.float64 ,'alphat09': np.float64 }, \
              names =['time_plot','alphat01' ,'alphat02','alphat03','alphat04','alphat05','alphat06','alphat07','alphat08','alphat09'], header =0)        
    df_epsilon_probe = pd.read_csv(path_probe + '0/epsilon.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'epsilon01': np.float64  ,'epsilon02': np.float64 ,'epsilon03': np.float64 ,'epsilon04': np.float64 ,'epsilon05': np.float64 , \
             'epsilon06': np.float64 ,'epsilon07': np.float64 ,'epsilon08': np.float64 ,'epsilon09': np.float64 }, \
              names =['time_plot','epsilon01' ,'epsilon02','epsilon03','epsilon04','epsilon05','epsilon06','epsilon07','epsilon08','epsilon09'], header =0)                        
    df_nut_probe = pd.read_csv(path_probe + '0/nut.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'nut01': np.float64  ,'nut02': np.float64 ,'nut03': np.float64 ,'nut04': np.float64 ,'nut05': np.float64 , \
             'nut06': np.float64 ,'nut07': np.float64 ,'nut08': np.float64 ,'nut09': np.float64 }, \
              names =['time_plot','nut01' ,'nut02','nut03','nut04','nut05','nut06','nut07','nut08','nut09'], header =0)        
    df_Ux_probe = pd.read_csv(path_probe + '0/Ux.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'Ux01': np.float64  ,'Ux02': np.float64 ,'Ux03': np.float64 ,'Ux04': np.float64 ,'Ux05': np.float64 , \
             'Ux06': np.float64 ,'Ux07': np.float64 ,'Ux08': np.float64 ,'Ux09': np.float64 }, \
              names =['time_plot','Ux01' ,'Ux02','Ux03','Ux04','Ux05','Ux06','Ux07','Ux08','Ux09'], header =0)
    df_Uy_probe = pd.read_csv(path_probe + '0/Uy.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'Uy01': np.float64  ,'Uy02': np.float64 ,'Uy03': np.float64 ,'Uy04': np.float64 ,'Uy05': np.float64 , \
             'Uy06': np.float64 ,'Uy07': np.float64 ,'Uy08': np.float64 ,'Uy09': np.float64 }, \
              names =['time_plot','Uy01' ,'Uy02','Uy03','Uy04','Uy05','Uy06','Uy07','Uy08','Uy09'], header =0)
    df_Uz_probe = pd.read_csv(path_probe + '0/Uz.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'Uz01': np.float64  ,'Uz02': np.float64 ,'Uz03': np.float64 ,'Uz04': np.float64 ,'Uz05': np.float64 , \
             'Uz06': np.float64 ,'Uz07': np.float64 ,'Uz08': np.float64 ,'Uz09': np.float64 }, \
              names =['time_plot','Uz01' ,'Uz02','Uz03','Uz04','Uz05','Uz06','Uz07','Uz08','Uz09'], header =0)        
    df_p_probe = pd.read_csv(path_probe + '0/p.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'p01': np.float64  ,'p02': np.float64 ,'p03': np.float64 ,'p04': np.float64 ,'p05': np.float64 , \
             'p06': np.float64 ,'p07': np.float64 ,'p08': np.float64 ,'p09': np.float64  }, \
              names =['time_plot','p01' ,'p02','p03','p04','p05','p06','p07','p08','p09'], header =0)
    df_p_rgh_probe = pd.read_csv(path_probe + '0/p_rgh.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'p_rgh01': np.float64  ,'p_rgh02': np.float64 ,'p_rgh03': np.float64 ,'p_rgh04': np.float64 ,'p_rgh05': np.float64 , \
             'p_rgh06': np.float64 ,'p_rgh07': np.float64 ,'p_rgh08': np.float64 ,'p_rgh09': np.float64  }, \
              names =['time_plot','p_rgh01' ,'p_rgh02','p_rgh03','p_rgh04','p_rgh05','p_rgh06','p_rgh07','p_rgh08','p_rgh09'], header =0)
    df_q_probe = pd.read_csv(path_probe + '0/q.csv',  comment='#', sep='\t' , index_col=0, \
              dtype ={'time_plot': np.float64 ,'q01': np.float64  ,'q02': np.float64 ,'q03': np.float64 ,'q04': np.float64 ,'q05': np.float64 , \
             'q06': np.float64 ,'q07': np.float64 ,'q08': np.float64 ,'q09': np.float64  }, \
              names =['time_plot','q01' ,'q02','q03','q04','q05','q06','q07','q08','q09'], header =0)
              
    df_probe = pd.concat([df_T_probe,df_k_probe,df_epsilon_probe,df_nut_probe,df_Ux_probe,df_Uy_probe,df_Uz_probe, df_p_probe,df_p_rgh_probe,df_q_probe,df_alphat_probe], axis=1)              
    
    return df_probe
# ******************************************************************************************
def plotting_sample_T(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_T',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['T_t0']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t1']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t2']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t3']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t4']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t5']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t6']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t7']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t8']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t9']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t10']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t11']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t12']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t13']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t14']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t15']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t16']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t17']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t18']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t19']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t20']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['T_t21']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t22']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['T_t23']-273.15, df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        print df_to_plot.index[0]
        print df_to_plot.index 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t0']-273.15, label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t1']-273.15, label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t2']-273.15, label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t3']-273.15, label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t4']-273.15, label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t5']-273.15, label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t6']-273.15, label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t7']-273.15, label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t8']-273.15, label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t9']-273.15, label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t10']-273.15, label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t11']-273.15, label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t12']-273.15, label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t13']-273.15, label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t14']-273.15, label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t15']-273.15, label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t16']-273.15, label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t17']-273.15, label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t18']-273.15, label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t19']-273.15, label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t20']-273.15, label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t21']-273.15, label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t22']-273.15, label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['T_t23']-273.15, label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample T -------------- > Plotted......"
    
    return
# ******************************************************************************************
def plotting_sample_alphat(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_alphat',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \alpha_t[m^2 s^{-1}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['alphat_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7 $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['alphat_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['alphat_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \alpha_t [m^2 s^{-1}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t14'], label= r'$t_{14} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['alphat_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample alphat -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_epsilon(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_epsilon',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon [m^2 s^{-3}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['epsilon_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0}$',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['epsilon_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['epsilon_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon [m^2 s^{-3}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t0'], label= r'$t_{0}$',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['epsilon_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample epsilon -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_k(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_k',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k [m^2 s^{-2}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['k_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['k_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['k_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k [m^2 s^{-2}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t8'], label= r'$t_{8}$',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['k_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample k -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_nut(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_nut',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \nu_t [m^2 s^{-1}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['nut_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['nut_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['nut_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \nu_t [m^2 s^{-1}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['nut_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample nut -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_p(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_p',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm p [m^2 s^{-2}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['p_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['p_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm p [m^2 s^{-2}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t19'], label= r'$t_{19} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample p -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_p_rgh(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_p_rgh',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm p_d [m^2 s^{-2}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['p_rgh_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['p_rgh_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['p_rgh_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm p_d [m^2 s^{-2}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['p_rgh_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample p_rgh -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_q(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 45
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_q',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q [kg kg^{-1}]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['q_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12}$',linewidth=line_width)
        #plt.plot(df_to_plot['q_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['q_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['q_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q [kg kg^{-1}]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t10'], label= r'$t_{11} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t19'], label= r'$t_{19}$',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['q_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample q -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_Ux(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_Ux',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U_x [m/s]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['Ux_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['Ux_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['Ux_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U_x [m/s]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Ux_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample Ux -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_Uy(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_Uy',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U_y [m/s]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['Uy_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['Uy_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uy_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U_y [m/s]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t2'], label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t19'], label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample Uy -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_Uz(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_Uz',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U_z [m/s]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot['Uz_t0'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t1'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t2'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t3'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t4'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t5'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t6'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t7'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t8'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t9'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t10'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t11'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t12'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t13'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t14'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t15'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t16'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t17'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t18'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t19'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t20'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(df_to_plot['Uz_t21'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t22'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(df_to_plot['Uz_t23'], df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U_z [m/s]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uy_t0'], label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t1'], label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t2'], label= r'$t_{2} $',linewidth=line_width)
       # plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t3'], label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t4'], label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t5'], label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t6'], label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t7'], label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t8'], label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t9'], label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t10'], label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t11'], label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t12'], label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t13'], label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t14'], label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t15'], label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t16'], label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t17'], label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t18'], label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t19'], label= r'$t_{19}$',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t20'], label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t21'], label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t22'], label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],df_to_plot['Uz_t23'], label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample Uz -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_U(df_samples_case,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    pylab.figure('simulated_sample_p',figsize=figure_size)
    pylab.clf()
    if (ax=='Z'):
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U [m/s]}$',size=x_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t0'],2)+np.power(df_to_plot['Uy_t0'],2)+np.power(df_to_plot['Uz_t0'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{0}$',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t1'],2)+np.power(df_to_plot['Uy_t1'],2)+np.power(df_to_plot['Uz_t1'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{1} $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t2'],2)+np.power(df_to_plot['Uy_t2'],2)+np.power(df_to_plot['Uz_t2'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t3'],2)+np.power(df_to_plot['Uy_t3'],2)+np.power(df_to_plot['Uz_t3'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t4'],2)+np.power(df_to_plot['Uy_t4'],2)+np.power(df_to_plot['Uz_t4'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t5'],2)+np.power(df_to_plot['Uy_t5'],2)+np.power(df_to_plot['Uz_t5'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t6'],2)+np.power(df_to_plot['Uy_t6'],2)+np.power(df_to_plot['Uz_t6'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t7'],2)+np.power(df_to_plot['Uy_t7'],2)+np.power(df_to_plot['Uz_t7'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{7} $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t8'],2)+np.power(df_to_plot['Uy_t8'],2)+np.power(df_to_plot['Uz_t8'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t9'],2)+np.power(df_to_plot['Uy_t9'],2)+np.power(df_to_plot['Uz_t9'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t10'],2)+np.power(df_to_plot['Uy_t10'],2)+np.power(df_to_plot['Uz_t10'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t11'],2)+np.power(df_to_plot['Uy_t11'],2)+np.power(df_to_plot['Uz_t11'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{11} $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t12'],2)+np.power(df_to_plot['Uy_t12'],2)+np.power(df_to_plot['Uz_t12'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t13'],2)+np.power(df_to_plot['Uy_t13'],2)+np.power(df_to_plot['Uz_t13'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t14'],2)+np.power(df_to_plot['Uy_t14'],2)+np.power(df_to_plot['Uz_t14'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{14} $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t15'],2)+np.power(df_to_plot['Uy_t15'],2)+np.power(df_to_plot['Uz_t15'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t16'],2)+np.power(df_to_plot['Uy_t16'],2)+np.power(df_to_plot['Uz_t16'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t17'],2)+np.power(df_to_plot['Uy_t17'],2)+np.power(df_to_plot['Uz_t17'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t18'],2)+np.power(df_to_plot['Uy_t18'],2)+np.power(df_to_plot['Uz_t18'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{18} $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t19'],2)+np.power(df_to_plot['Uy_t19'],2)+np.power(df_to_plot['Uz_t19'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t20'],2)+np.power(df_to_plot['Uy_t20'],2)+np.power(df_to_plot['Uz_t20'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{20} $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(df_to_plot['Ux_t21'],2)+np.power(df_to_plot['Uy_t21'],2)+np.power(df_to_plot['Uz_t21'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{21} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t22'],2)+np.power(df_to_plot['Uy_t22'],2)+np.power(df_to_plot['Uz_t22'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{22} $',linewidth=line_width)
        #plt.plot(np.sqrt(np.power(df_to_plot['Ux_t23'],2)+np.power(df_to_plot['Uy_t23'],2)+np.power(df_to_plot['Uz_t23'],2)), df_to_plot.index.to_series()-df_to_plot.index[0],label= r'$t_{23} $',linewidth=line_width)
    if (ax=='X'):
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U [m/s]}$',size=y_lable_size)
        # REVERSE THE Y-AXIS
        #plt.gca().invert_yaxis()
        #ax = plt.gca()
        #ax.set_ylim(ax.get_ylim()[::-1])
        #plt.show()
        # Plotting
        df_to_plot = df_samples_case
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t0'],2)+np.power(df_to_plot['Uy_t0'],2)+np.power(df_to_plot['Uz_t0'],2)), label= r'$t_{0} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t1'],2)+np.power(df_to_plot['Uy_t1'],2)+np.power(df_to_plot['Uz_t1'],2)), label= r'$t_{1} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t2'],2)+np.power(df_to_plot['Uy_t2'],2)+np.power(df_to_plot['Uz_t2'],2)), label= r'$t_{2} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t3'],2)+np.power(df_to_plot['Uy_t3'],2)+np.power(df_to_plot['Uz_t3'],2)), label= r'$t_{3} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t4'],2)+np.power(df_to_plot['Uy_t4'],2)+np.power(df_to_plot['Uz_t4'],2)), label= r'$t_{4} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t5'],2)+np.power(df_to_plot['Uy_t5'],2)+np.power(df_to_plot['Uz_t5'],2)), label= r'$t_{5} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t6'],2)+np.power(df_to_plot['Uy_t6'],2)+np.power(df_to_plot['Uz_t6'],2)), label= r'$t_{6} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t7'],2)+np.power(df_to_plot['Uy_t7'],2)+np.power(df_to_plot['Uz_t7'],2)), label= r'$t_{7} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t8'],2)+np.power(df_to_plot['Uy_t8'],2)+np.power(df_to_plot['Uz_t8'],2)), label= r'$t_{8} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t9'],2)+np.power(df_to_plot['Uy_t9'],2)+np.power(df_to_plot['Uz_t9'],2)), label= r'$t_{9} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t10'],2)+np.power(df_to_plot['Uy_t10'],2)+np.power(df_to_plot['Uz_t10'],2)), label= r'$t_{10} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t11'],2)+np.power(df_to_plot['Uy_t11'],2)+np.power(df_to_plot['Uz_t11'],2)), label= r'$t_{11} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t12'],2)+np.power(df_to_plot['Uy_t12'],2)+np.power(df_to_plot['Uz_t12'],2)), label= r'$t_{12} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t13'],2)+np.power(df_to_plot['Uy_t13'],2)+np.power(df_to_plot['Uz_t13'],2)), label= r'$t_{13} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t14'],2)+np.power(df_to_plot['Uy_t14'],2)+np.power(df_to_plot['Uz_t14'],2)), label= r'$t_{14} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t15'],2)+np.power(df_to_plot['Uy_t15'],2)+np.power(df_to_plot['Uz_t15'],2)), label= r'$t_{15} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t16'],2)+np.power(df_to_plot['Uy_t16'],2)+np.power(df_to_plot['Uz_t16'],2)), label= r'$t_{16} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t17'],2)+np.power(df_to_plot['Uy_t17'],2)+np.power(df_to_plot['Uz_t17'],2)), label= r'$t_{17} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t18'],2)+np.power(df_to_plot['Uy_t18'],2)+np.power(df_to_plot['Uz_t18'],2)), label= r'$t_{18} $',linewidth=line_width)
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t19'],2)+np.power(df_to_plot['Uy_t19'],2)+np.power(df_to_plot['Uz_t19'],2)), label= r'$t_{19} $',linewidth=line_width)
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t20'],2)+np.power(df_to_plot['Uy_t20'],2)+np.power(df_to_plot['Uz_t20'],2)), label= r'$t_{20} $',linewidth=line_width) 
        plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t21'],2)+np.power(df_to_plot['Uy_t21'],2)+np.power(df_to_plot['Uz_t21'],2)), label= r'$t_{21} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t22'],2)+np.power(df_to_plot['Uy_t22'],2)+np.power(df_to_plot['Uz_t22'],2)), label= r'$t_{22} $',linewidth=line_width) 
        #plt.plot(df_to_plot.index.to_series()-df_to_plot.index[0],np.sqrt(np.power(df_to_plot['Ux_t23'],2)+np.power(df_to_plot['Uy_t23'],2)+np.power(df_to_plot['Uz_t23'],2)), label= r'$t_{23} $',linewidth=line_width) 
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample.tif',bbox_inches='tight')
    plt.show()
    print "sample U -------------- > Plotted......"
    return
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
def plotting_probe_T(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_T01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['T01']-273.15, color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T02']-273.15, color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T03']-273.15, color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe T01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_T02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['T04']-273.15, color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T05']-273.15, color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T06']-273.15, color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe T02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_T03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['T07']-273.15, color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T08']-273.15, color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T09']-273.15, color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe T03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_T04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['T01']-273.15, color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T04']-273.15, color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T07']-273.15, color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe T04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_T05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['T02']-273.15, color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T05']-273.15, color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T08']-273.15, color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe T05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_T06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['T03']-273.15, color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T06']-273.15, color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['T09']-273.15, color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/T06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe T06 -------------- > Plotted......"
    return
    #*****************************************************************************************
def plotting_probe_epsilon(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_epsilon01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \varepsilon[m^2s^{-3}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe epsilon01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_epsilon02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \varepsilon[m^2s^{-3}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe epsilon02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_epsilon03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \varepsilon[m^2s^{-3}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe epsilon03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_epsilon04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \varepsilon[m^2s^{-3}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe epsilon04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_epsilon05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \varepsilon[m^2s^{-3}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe epsilon05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_epsilon06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \varepsilon[m^2s^{-3}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['epsilon09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe epsilon06 -------------- > Plotted......"
    return
    #*****************************************************************************************
def plotting_probe_alphat(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_alphat01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \alpha_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe alphat01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_alphat02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \alpha_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe alphat02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_alphat03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \alpha_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe alphat03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_alphat04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \alpha_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe alphat04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_alphat05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \alpha_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe alphat05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_alphat06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \alpha_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['alphat09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/alphat06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe alphat06 -------------- > Plotted......"
    return
    #*****************************************************************************************
def plotting_probe_k(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_k01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm k[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['k01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe k01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_k02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm k [m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['k04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe k02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_k03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm k[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['k07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe k03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_k04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm k[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['k01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe k04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_k05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm k[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['k02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe k05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_k06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm k[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['k03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['k09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/k06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe k06 -------------- > Plotted......"
    return
    #*****************************************************************************************
def plotting_probe_nut(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------        
    pylab.figure('simulated_probe_nut01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \nu_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['nut01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe nut01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_nut02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \nu_t [m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['nut04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe nut02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_nut03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \nu_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['nut07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe nut03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_nut04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \nu_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['nut01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe nut04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_nut05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \nu_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['nut02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe nut05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_nut06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm \nu_t[m^2s^{-1}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['nut03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['nut09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/nut06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe nut06 -------------- > Plotted......"  
    return
    #*****************************************************************************************
def plotting_probe_p(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_p01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_p02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p [m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_p03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_p04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_p05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_p06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p06 -------------- > Plotted......"   
    return
    #*****************************************************************************************
def plotting_probe_p_rgh(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_p_rgh01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p_d[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p_rgh01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_p_rgh02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p_d[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p_rgh02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_p_rgh03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p_d[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p_rgh03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_p_rgh04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p_d[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p_rgh04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_p_rgh05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p_d[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p_rgh05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_p_rgh06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm p_d[m^2s^{-2}]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['p_rgh09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/p_rgh06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe p_rgh06 -------------- > Plotted......"    
    return
    #*****************************************************************************************
def plotting_probe_q(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_q01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['q01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe q01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_q02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['q04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe q02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_q03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['q07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe q03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_q04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['q01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe q04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_q05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['q02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe q05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_q06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['q03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['q09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/q06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe q06 -------------- > Plotted......"  
    return
    #*****************************************************************************************
def plotting_probe_Ux(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------    
    pylab.figure('simulated_probe_Ux01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Ux[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Ux01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_Ux02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Ux[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Ux02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_Ux03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Ux[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Ux03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_Ux04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Ux[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Ux04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_Ux05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Ux[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Ux05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_Ux06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Ux[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Ux09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Ux06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Ux06 -------------- > Plotted......"  
    return
    #*****************************************************************************************
def plotting_probe_Uy(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------        
    pylab.figure('simulated_probe_Uy01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uy[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uy01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_Uy02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uy[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uy02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_Uy03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uy[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uy03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_Uy04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uy[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uy04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_Uy05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uy[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uy05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_Uy06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uy[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uy09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uy06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uy06 -------------- > Plotted......"  
    return
    #*****************************************************************************************
def plotting_probe_Uz(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------       
    pylab.figure('simulated_probe_Uz01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uz[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz01'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz02'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz03'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uz01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_Uz02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uz[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz04'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz05'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz06'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uz02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_Uz03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uz[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz07'], color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz08'], color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz09'], color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uz03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_Uz04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uz[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz01'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz04'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz07'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uz04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_Uz05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uz[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz02'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz05'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz08'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uz05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_Uz06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm Uz[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz03'], color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz06'], color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, df_to_plot['Uz09'], color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/Uz06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe Uz06 -------------- > Plotted......"  
    return
    #*****************************************************************************************
def plotting_probe_U(df_probes_case):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size_probe=(20* 1.618, 20)
    line_width_probe = 3.0
    df_to_plot = df_probes_case
    #-------------------------        
    pylab.figure('simulated_probe_U01',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
   #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux01'],2)+np.power(df_to_plot['Uy01'],2)+np.power(df_to_plot['Uz01'],2)), color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux02'],2)+np.power(df_to_plot['Uy02'],2)+np.power(df_to_plot['Uz02'],2)), color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux03'],2)+np.power(df_to_plot['Uy03'],2)+np.power(df_to_plot['Uz03'],2)), color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U01_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe U01 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_U02',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux04'],2)+np.power(df_to_plot['Uy04'],2)+np.power(df_to_plot['Uz04'],2)), color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux05'],2)+np.power(df_to_plot['Uy05'],2)+np.power(df_to_plot['Uz05'],2)), color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux06'],2)+np.power(df_to_plot['Uy06'],2)+np.power(df_to_plot['Uz06'],2)), color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U02_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe U02 -------------- > Plotted......"
    #-----------------------------------------------------------------
    pylab.figure('simulated_probe_U03',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux07'],2)+np.power(df_to_plot['Uy07'],2)+np.power(df_to_plot['Uz07'],2)), color='r',label= r'$z=0.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux08'],2)+np.power(df_to_plot['Uy08'],2)+np.power(df_to_plot['Uz08'],2)), color='b',label= r'$z=2.0$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux09'],2)+np.power(df_to_plot['Uy09'],2)+np.power(df_to_plot['Uz09'],2)), color='g',label= r'$z=10.0$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U03_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe U03 -------------- > Plotted......"
    #-----------------------------------------------------------------    
    pylab.figure('simulated_probe_U04',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux01'],2)+np.power(df_to_plot['Uy01'],2)+np.power(df_to_plot['Uz01'],2)), color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux04'],2)+np.power(df_to_plot['Uy04'],2)+np.power(df_to_plot['Uz04'],2)), color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux07'],2)+np.power(df_to_plot['Uy07'],2)+np.power(df_to_plot['Uz07'],2)), color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U04_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe U04 -------------- > Plotted......"
    #-----------------------------------------------------------------        
    pylab.figure('simulated_probe_U05',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux02'],2)+np.power(df_to_plot['Uy02'],2)+np.power(df_to_plot['Uz02'],2)), color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux05'],2)+np.power(df_to_plot['Uy05'],2)+np.power(df_to_plot['Uz05'],2)), color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux08'],2)+np.power(df_to_plot['Uy08'],2)+np.power(df_to_plot['Uz08'],2)), color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U05_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe U05 -------------- > Plotted......"
    #-----------------------------------------------------------------          
    pylab.figure('simulated_probe_U06',figsize=figure_size_probe)
    pylab.clf()
    #pylab.subplot(341)
    pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
    pylab.xlabel(r'${\rm Time[hr]}$',size=x_lable_size)
    #df_to_plot = df_probes_case
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux03'],2)+np.power(df_to_plot['Uy03'],2)+np.power(df_to_plot['Uz03'],2)), color='r',label= r'$x=500$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux06'],2)+np.power(df_to_plot['Uy06'],2)+np.power(df_to_plot['Uz06'],2)), color='b',label= r'$x=1000$',linewidth=line_width_probe)
    plt.plot(df_to_plot.index/3600, np.sqrt(np.power(df_to_plot['Ux09'],2)+np.power(df_to_plot['Uy09'],2)+np.power(df_to_plot['Uz09'],2)), color='g',label= r'$x=1500$',linewidth=line_width_probe)
    pylab.gcf().autofmt_xdate()
    plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
    plt.legend(loc='best',prop={'size':legend_size})
    plt.gcf().autofmt_xdate()
    plt.savefig(reading_input.save_graphs +'/ABL_results/U06_probe.tif',bbox_inches='tight')
    plt.show()
    print "probe U06 -------------- > Plotted......"  
    return
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#----------------------------------------------------------------
def plotting_sample_T_01(s2,s3,s4,s5,s6,s7,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='Z'):
        pylab.figure('simulated_sample_T_t1',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t1']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t1']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t1']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t1']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t1']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t1']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t1']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_T_t2',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t2']-273.15, s1.index,label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t2']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t2']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t2']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t2']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t2']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t2']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_T_t8',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t8']-273.15, s1.index.to_series()-s.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t8']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t8']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t8']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t8']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t8']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t8']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_T_t12',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t12']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t12']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t12']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t12']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t12']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t12']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t12']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_T_t15',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t15']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t15']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t15']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t15']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t15']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t15']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t15']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_T_t19',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t19']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t19']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t19']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t19']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t19']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t19']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t19']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_T_t21',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm Temperature[C]}$',size=x_lable_size)
        #plt.plot(s1['T_t21']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['T_t21']-273.15, s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['T_t21']-273.15, s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['T_t21']-273.15, s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['T_t21']-273.15, s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['T_t21']-273.15, s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['T_t21']-273.15, s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/T_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample T_t21 -------------- > Plotted......"           
    print "sample T_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_epsilon_01(s2,s3,s4,s5,s6,s7,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='Z'):
        pylab.figure('simulated_sample_epsilon_t1',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t1']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t1'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t1'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t1'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t1'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t1'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t1'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilon_t2',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t2'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t2'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t2'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t2'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t2'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t2'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t2'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilon_t8',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t8'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t8'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t8'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t8'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t8'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t8'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t8'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilon_t12',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t12'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t12'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t12'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t12'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t12'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t12'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t12'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilon_t15',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t15'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t15'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t15'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t15'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t15'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t15'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t15'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilon_t19',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t19'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t19'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t19'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t19'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t19'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t19'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t19'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilon_t21',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=x_lable_size)
        #plt.plot(s1['epsilon_t21'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['epsilon_t21'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['epsilon_t21'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['epsilon_t21'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['epsilon_t21'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['epsilon_t21'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['epsilon_t21'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilon_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilon_t21 -------------- > Plotted......"           
    print "sample epsilon_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_k_01(s2,s3,s4,s5,s6,s7,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='Z'):
        pylab.figure('simulated_sample_k_t1',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t1'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t1'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t1'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t1'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t1'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t1'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t1'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_k_t2',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t2'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t2'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t2'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t2'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t2'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t2'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t2'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_k_t8',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t8'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t8'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t8'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t8'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t8'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t8'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t8'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_k_t12',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t12'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t12'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t12'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t12'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t12'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t12'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t12'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_k_t15',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t15'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t15'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t15'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t15'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t15'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t15'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t15'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_k_t19',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t19'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t19'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t19'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t19'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t19'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t19'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t19'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_k_t21',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm k[m^2 s^{-2}]}$',size=x_lable_size)
        #plt.plot(s1['k_t21'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['k_t21'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['k_t21'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['k_t21'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['k_t21'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['k_t21'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['k_t21'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/k_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample k_t21 -------------- > Plotted......"           
    print "sample k_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_q_01(s2,s3,s4,s5,s6,s7,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='Z'):
        pylab.figure('simulated_sample_q_t1',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t1']-273.15, s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t1'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t1'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t1'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t1'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t1'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t1'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_q_t2',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t2'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t2'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t2'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t2'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t2'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t2'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t2'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_q_t8',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t8'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t8'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t8'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t8'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t8'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t8'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t8'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_q_t12',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t12'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t12'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t12'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t12'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t12'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t12'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t12'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_q_t15',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t15'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t15'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t15'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t15'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t15'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t15'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t15'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_q_t19',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t19'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t19'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t19'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t19'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t19'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t19'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t19'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_q_t21',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm q[kg/kg]}$',size=x_lable_size)
        #plt.plot(s1['q_t21'], s1.index.to_series()-s1.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(s2['q_t21'], s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(s3['q_t21'], s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(s4['q_t21'], s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(s5['q_t21'], s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(s6['q_t21'], s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(s7['q_t21'], s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/q_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample q_t21 -------------- > Plotted......"           
    print "sample q_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_U_01(s2,s3,s4,s5,s6,s7,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='Z'):
        pylab.figure('simulated_sample_U_t1',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t1']-273.15, s1.index - z_origin,label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t1'],2)+np.power(s2['Uy_t1'],2)+np.power(s2['Uz_t1'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t1'],2)+np.power(s3['Uy_t1'],2)+np.power(s3['Uz_t1'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t1'],2)+np.power(s4['Uy_t1'],2)+np.power(s4['Uz_t1'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t1'],2)+np.power(s5['Uy_t1'],2)+np.power(s5['Uz_t1'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t1'],2)+np.power(s6['Uy_t1'],2)+np.power(s6['Uz_t1'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t1'],2)+np.power(s7['Uy_t1'],2)+np.power(s7['Uz_t1'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_U_t2',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t2'], s1.index - z_origin,label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t2'],2)+np.power(s2['Uy_t2'],2)+np.power(s2['Uz_t2'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t2'],2)+np.power(s3['Uy_t2'],2)+np.power(s3['Uz_t2'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t2'],2)+np.power(s4['Uy_t2'],2)+np.power(s4['Uz_t2'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t2'],2)+np.power(s5['Uy_t2'],2)+np.power(s5['Uz_t2'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t2'],2)+np.power(s6['Uy_t2'],2)+np.power(s6['Uz_t2'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t2'],2)+np.power(s7['Uy_t2'],2)+np.power(s7['Uz_t2'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_U_t8',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t8'], s1.index - z_origin,label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t8'],2)+np.power(s2['Uy_t8'],2)+np.power(s2['Uz_t8'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t8'],2)+np.power(s3['Uy_t8'],2)+np.power(s3['Uz_t8'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t8'],2)+np.power(s4['Uy_t8'],2)+np.power(s4['Uz_t8'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t8'],2)+np.power(s5['Uy_t8'],2)+np.power(s5['Uz_t8'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t8'],2)+np.power(s6['Uy_t8'],2)+np.power(s6['Uz_t8'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t8'],2)+np.power(s7['Uy_t8'],2)+np.power(s7['Uz_t8'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_U_t12',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t12'], s1.index - z_origin,label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t12'],2)+np.power(s2['Uy_t12'],2)+np.power(s2['Uz_t12'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t12'],2)+np.power(s3['Uy_t12'],2)+np.power(s3['Uz_t12'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t12'],2)+np.power(s4['Uy_t12'],2)+np.power(s4['Uz_t12'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t12'],2)+np.power(s5['Uy_t12'],2)+np.power(s5['Uz_t12'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t12'],2)+np.power(s6['Uy_t12'],2)+np.power(s6['Uz_t12'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t12'],2)+np.power(s7['Uy_t12'],2)+np.power(s7['Uz_t12'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_U_t15',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t15'], s1.index.to_series()-s.index[0],label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t15'],2)+np.power(s2['Uy_t15'],2)+np.power(s2['Uz_t15'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t15'],2)+np.power(s3['Uy_t15'],2)+np.power(s3['Uz_t15'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t15'],2)+np.power(s4['Uy_t15'],2)+np.power(s4['Uz_t15'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t15'],2)+np.power(s5['Uy_t15'],2)+np.power(s5['Uz_t15'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t15'],2)+np.power(s6['Uy_t15'],2)+np.power(s6['Uz_t15'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t15'],2)+np.power(s7['Uy_t15'],2)+np.power(s7['Uz_t15'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_U_t19',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t19'], s1.index - z_origin,label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t19'],2)+np.power(s2['Uy_t19'],2)+np.power(s2['Uz_t19'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t19'],2)+np.power(s3['Uy_t19'],2)+np.power(s3['Uz_t19'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t19'],2)+np.power(s4['Uy_t19'],2)+np.power(s4['Uz_t19'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t19'],2)+np.power(s5['Uy_t19'],2)+np.power(s5['Uz_t19'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t19'],2)+np.power(s6['Uy_t19'],2)+np.power(s6['Uz_t19'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t19'],2)+np.power(s7['Uy_t19'],2)+np.power(s7['Uz_t19'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_U_t21',figsize=figure_size)
        pylab.clf()
        pylab.ylabel(r'${\rm Height[m]}$',size=y_lable_size)
        pylab.xlabel(r'${\rm U[m/s]}$',size=x_lable_size)
        #plt.plot(s1['U_t21'], s1.index - z_origin,label= r'$z_1 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s2['Ux_t21'],2)+np.power(s2['Uy_t21'],2)+np.power(s2['Uz_t21'],2)), s2.index.to_series()-s2.index[0],label= r'$z_2 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s3['Ux_t21'],2)+np.power(s3['Uy_t21'],2)+np.power(s3['Uz_t21'],2)), s3.index.to_series()-s3.index[0],label= r'$z_3 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s4['Ux_t21'],2)+np.power(s4['Uy_t21'],2)+np.power(s4['Uz_t21'],2)), s4.index.to_series()-s4.index[0],label= r'$z_4 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s5['Ux_t21'],2)+np.power(s5['Uy_t21'],2)+np.power(s5['Uz_t21'],2)), s5.index.to_series()-s5.index[0],label= r'$z_5 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s6['Ux_t21'],2)+np.power(s6['Uy_t21'],2)+np.power(s6['Uz_t21'],2)), s6.index.to_series()-s6.index[0],label= r'$z_6 $',linewidth=line_width)
        plt.plot(np.sqrt(np.power(s7['Ux_t21'],2)+np.power(s7['Uy_t21'],2)+np.power(s7['Uz_t21'],2)), s7.index.to_series()-s7.index[0],label= r'$z_7 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/U_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample U_t21 -------------- > Plotted......"           
    print "sample U_ti -------------- > Plotted......"
    return
# ******************************************************************************************
#--------------------------------------------------------------------------------------------
def plotting_sample_T_02(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='X'):
        pylab.figure('simulated_sample_Tx01_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t1']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t1']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t1']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t1']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t1']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t1']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t1']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t1']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t1']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t1']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t1']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t1']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx01_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t2']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t2']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t2']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t2']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx01_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx01_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t2']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t2']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t2']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t2']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t2 -------------- > Plotted......"
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t2']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t2']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t2']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t2']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx01_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t8']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t8']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t8']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t8']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx01_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx01_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t8']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t8']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t8']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t8']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t8 -------------- > Plotted......"        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t8']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t8']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t8']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t8']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx01_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t12']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t12']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t12']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t12']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx01_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx01_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t12']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t12']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t12']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t12']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t12']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t12']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t12']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t12']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx01_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t15']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t15']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t15']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t15']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx01_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx01_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t15']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t15']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t15']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t15']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t15']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t15']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t15']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t15']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx01_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t19']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t19']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t19']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t19']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx01_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx01_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t19']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t19']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t19']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t19']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t19 -------------- > Plotted......"      
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t19']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t19']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t19']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t19']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx01_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['T_t21']-273.15, label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['T_t21']-273.15,label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['T_t21']-273.15, label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['T_t21']-273.15,label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx01_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx01_t21 -------------- > Plotted......"           
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx02_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['T_t21']-273.15, label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['T_t21']-273.15, label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['T_t21']-273.15, label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['T_t21']-273.15, label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx02_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx02_t21 -------------- > Plotted......"    
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Tx03_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm Temperature[C]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['T_t21']-273.15, label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['T_t21']-273.15, label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['T_t21']-273.15, label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['T_t21']-273.15, label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Tx03_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample Tx03_t21 -------------- > Plotted......"         
    print "sample Tx0i_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_epsilon_02(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='X'):
        pylab.figure('simulated_sample_epsilonx01_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t1'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t1'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t1'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t1'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t1'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t1'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t1'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t1'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t1'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t1'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t1'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t1'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx01_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t2'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t2'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t2'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t2'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx01_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx01_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t2'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t2'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t2'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t2'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t2 -------------- > Plotted......"
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t2'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t2'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t2'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t2'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx01_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t8'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t8'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t8'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t8'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx01_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx01_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t8'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t8'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t8'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t8'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t8 -------------- > Plotted......"        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t8'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t8'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t8'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t8'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx01_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t12'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t12'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t12'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t12'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx01_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx01_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t12'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t12'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t12'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t12'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t12'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t12'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t12'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t12'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx01_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t15'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t15'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t15'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t15'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx01_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx01_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t15'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t15'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t15'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t15'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t15'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t15'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t15'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t15'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx01_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t19'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t19'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t19'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t19'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx01_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx01_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t19'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t19'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t19'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t19'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t19 -------------- > Plotted......"      
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t19'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t19'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t19'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t19'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx01_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['epsilon_t21'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['epsilon_t21'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['epsilon_t21'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['epsilon_t21'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx01_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx01_t21 -------------- > Plotted......"           
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx02_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['epsilon_t21'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['epsilon_t21'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['epsilon_t21'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['epsilon_t21'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx02_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx02_t21 -------------- > Plotted......"    
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_epsilonx03_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm \varepsilon[m^2 s^{-3}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['epsilon_t21'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['epsilon_t21'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['epsilon_t21'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['epsilon_t21'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/epsilonx03_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample epsilonx03_t21 -------------- > Plotted......"         
    print "sample epsilonx0i_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_k_02(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='X'):
        pylab.figure('simulated_sample_kx01_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t1'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t1'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t1'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t1'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample kx_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t1'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t1'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t1'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t1'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t1'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t1'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t1'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t1'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx01_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t2'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t2'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t2'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t2'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx01_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample kx01_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t2'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t2'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t2'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t2'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t2 -------------- > Plotted......"
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t2'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t2'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t2'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t2'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx01_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t8'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t8'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t8'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t8'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx01_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample kx01_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t8'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t8'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t8'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t8'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t8 -------------- > Plotted......"        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t8'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t8'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t8'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t8'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx01_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t12'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t12'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t12'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t12'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx01_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample kx01_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t12'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t12'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t12'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t12'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t12'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t12'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t12'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t12'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx01_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t15'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t15'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t15'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t15'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx01_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample kx01_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t15'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t15'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t15'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t15'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t15'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t15'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t15'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t15'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx01_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t19'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t19'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t19'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t19'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx01_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample kx01_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t19'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t19'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t19'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t19'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t19 -------------- > Plotted......"      
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t19'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t19'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t19'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t19'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx01_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['k_t21'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['k_t21'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['k_t21'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['k_t21'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx01_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample kx01_t21 -------------- > Plotted......"           
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx02_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['k_t21'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['k_t21'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['k_t21'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['k_t21'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx02_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample kx02_t21 -------------- > Plotted......"    
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_kx03_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm k[m^2 s^{-2}]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['k_t21'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['k_t21'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['k_t21'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['k_t21'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/kx03_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample kx03_t21 -------------- > Plotted......"         
    print "sample kx0i_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_q_02(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='X'):
        pylab.figure('simulated_sample_qx01_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t1'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t1'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t1'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t1'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample qx_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t1'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t1'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t1'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t1'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t1'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t1'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t1'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t1'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx01_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t2'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t2'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t2'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t2'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx01_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample qx01_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t2'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t2'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t2'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t2'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t2 -------------- > Plotted......"
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t2'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t2'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t2'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t2'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx01_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t8'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t8'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t8'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t8'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx01_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample qx01_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t8'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t8'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t8'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t8'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t8 -------------- > Plotted......"        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t8'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t8'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t8'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t8'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx01_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t12'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t12'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t12'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t12'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx01_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample qx01_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t12'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t12'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t12'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t12'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t12'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t12'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t12'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t12'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx01_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t15'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t15'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t15'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t15'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx01_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample qx01_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t15'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t15'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t15'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t15'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t15'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t15'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t15'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t15'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx01_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t19'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t19'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t19'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t19'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx01_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample qx01_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t19'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t19'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t19'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t19'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t19 -------------- > Plotted......"      
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t19'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t19'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t19'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t19'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx01_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], s1['q_t21'], label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], s2['q_t21'],label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], s3['q_t21'], label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], s4['q_t21'],label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx01_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample qx01_t21 -------------- > Plotted......"           
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx02_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], s5['q_t21'], label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], s6['q_t21'], label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], s7['q_t21'], label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], s8['q_t21'], label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx02_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample qx02_t21 -------------- > Plotted......"    
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_qx03_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm q[kg/kg]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], s9['q_t21'], label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], s10['q_t21'], label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], s11['q_t21'], label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], s12['q_t21'], label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/qx03_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample qx03_t21 -------------- > Plotted......"         
    print "sample qx0i_ti -------------- > Plotted......"
    return
# ******************************************************************************************
def plotting_sample_U_02(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,ax):
    x_lable_size = 60
    y_lable_size = 60
    legend_size = 55
    #titel_size = 70
    figure_size=(20* 1.618, 20)
    line_width = 3.0
    if (ax=='X'):
        pylab.figure('simulated_sample_Ux01_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t1'],2)+np.power(s1['Uy_t1'],2)+np.power(s1['Uz_t1'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t1'],2)+np.power(s2['Uy_t1'],2)+np.power(s2['Uz_t1'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t1'],2)+np.power(s3['Uy_t1'],2)+np.power(s3['Uz_t1'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t1'],2)+np.power(s4['Uy_t1'],2)+np.power(s4['Uz_t1'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t1'],2)+np.power(s5['Uy_t1'],2)+np.power(s5['Uz_t1'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t1'],2)+np.power(s6['Uy_t1'],2)+np.power(s6['Uz_t1'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t1'],2)+np.power(s7['Uy_t1'],2)+np.power(s7['Uz_t1'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t1'],2)+np.power(s8['Uy_t1'],2)+np.power(s8['Uz_t1'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t1',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t1'],2)+np.power(s9['Uy_t1'],2)+np.power(s9['Uz_t1'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t1'],2)+np.power(s10['Uy_t1'],2)+np.power(s10['Uz_t1'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t1'],2)+np.power(s11['Uy_t1'],2)+np.power(s11['Uz_t1'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t1'],2)+np.power(s12['Uy_t1'],2)+np.power(s12['Uz_t1'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t1.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t1 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux01_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t2'],2)+np.power(s1['Uy_t2'],2)+np.power(s1['Uz_t2'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t2'],2)+np.power(s2['Uy_t2'],2)+np.power(s2['Uz_t2'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t2'],2)+np.power(s3['Uy_t2'],2)+np.power(s3['Uz_t2'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t2'],2)+np.power(s4['Uy_t2'],2)+np.power(s4['Uz_t2'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux01_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t2'],2)+np.power(s5['Uy_t2'],2)+np.power(s5['Uz_t2'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t2'],2)+np.power(s6['Uy_t2'],2)+np.power(s6['Uz_t2'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t2'],2)+np.power(s7['Uy_t2'],2)+np.power(s7['Uz_t2'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t2'],2)+np.power(s8['Uy_t2'],2)+np.power(s8['Uz_t2'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t2 -------------- > Plotted......"
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t2',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t2'],2)+np.power(s9['Uy_t2'],2)+np.power(s9['Uz_t2'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t2'],2)+np.power(s10['Uy_t2'],2)+np.power(s10['Uz_t2'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t2'],2)+np.power(s11['Uy_t2'],2)+np.power(s11['Uz_t2'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t2'],2)+np.power(s12['Uy_t2'],2)+np.power(s12['Uz_t2'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t2.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t2 -------------- > Plotted......"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux01_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t8'],2)+np.power(s1['Uy_t8'],2)+np.power(s1['Uz_t8'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t8'],2)+np.power(s2['Uy_t8'],2)+np.power(s2['Uz_t8'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t8'],2)+np.power(s3['Uy_t8'],2)+np.power(s3['Uz_t8'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t8'],2)+np.power(s4['Uy_t8'],2)+np.power(s4['Uz_t8'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux01_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t8'],2)+np.power(s5['Uy_t8'],2)+np.power(s5['Uz_t8'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t8'],2)+np.power(s6['Uy_t8'],2)+np.power(s6['Uz_t8'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t8'],2)+np.power(s7['Uy_t8'],2)+np.power(s7['Uz_t8'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t8'],2)+np.power(s8['Uy_t8'],2)+np.power(s8['Uz_t8'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t8 -------------- > Plotted......"        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t8',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t8'],2)+np.power(s9['Uy_t8'],2)+np.power(s9['Uz_t8'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t8'],2)+np.power(s10['Uy_t8'],2)+np.power(s10['Uz_t8'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t8'],2)+np.power(s11['Uy_t8'],2)+np.power(s11['Uz_t8'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t8'],2)+np.power(s12['Uy_t8'],2)+np.power(s12['Uz_t8'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t8.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t8 -------------- > Plotted......"        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux01_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t12'],2)+np.power(s1['Uy_t12'],2)+np.power(s1['Uz_t12'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t12'],2)+np.power(s2['Uy_t12'],2)+np.power(s2['Uz_t12'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t12'],2)+np.power(s3['Uy_t12'],2)+np.power(s3['Uz_t12'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t12'],2)+np.power(s4['Uy_t12'],2)+np.power(s4['Uz_t12'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux01_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t12'],2)+np.power(s5['Uy_t12'],2)+np.power(s5['Uz_t12'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t12'],2)+np.power(s6['Uy_t12'],2)+np.power(s6['Uz_t12'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t12'],2)+np.power(s7['Uy_t12'],2)+np.power(s7['Uz_t12'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t12'],2)+np.power(s8['Uy_t12'],2)+np.power(s8['Uz_t12'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t12',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t12'],2)+np.power(s9['Uy_t12'],2)+np.power(s9['Uz_t12'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t12'],2)+np.power(s10['Uy_t12'],2)+np.power(s10['Uz_t12'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t12'],2)+np.power(s11['Uy_t12'],2)+np.power(s11['Uz_t12'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t12'],2)+np.power(s12['Uy_t12'],2)+np.power(s12['Uz_t12'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t12.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t12 -------------- > Plotted......"                
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux01_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t15'],2)+np.power(s1['Uy_t15'],2)+np.power(s1['Uz_t15'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t15'],2)+np.power(s2['Uy_t15'],2)+np.power(s2['Uz_t15'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t15'],2)+np.power(s3['Uy_t15'],2)+np.power(s3['Uz_t15'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t15'],2)+np.power(s4['Uy_t15'],2)+np.power(s4['Uz_t15'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux01_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t15'],2)+np.power(s5['Uy_t15'],2)+np.power(s5['Uz_t15'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t15'],2)+np.power(s6['Uy_t15'],2)+np.power(s6['Uz_t15'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t15'],2)+np.power(s7['Uy_t15'],2)+np.power(s7['Uz_t15'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t15'],2)+np.power(s8['Uy_t15'],2)+np.power(s8['Uz_t15'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t15',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t15'],2)+np.power(s9['Uy_t15'],2)+np.power(s9['Uz_t15'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t15'],2)+np.power(s10['Uy_t15'],2)+np.power(s10['Uz_t15'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t15'],2)+np.power(s11['Uy_t15'],2)+np.power(s11['Uz_t15'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t15'],2)+np.power(s12['Uy_t15'],2)+np.power(s12['Uz_t15'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t15.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t15 -------------- > Plotted......"          
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux01_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t19'],2)+np.power(s1['Uy_t19'],2)+np.power(s1['Uz_t19'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t19'],2)+np.power(s2['Uy_t19'],2)+np.power(s2['Uz_t19'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t19'],2)+np.power(s3['Uy_t19'],2)+np.power(s3['Uz_t19'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t19'],2)+np.power(s4['Uy_t19'],2)+np.power(s4['Uz_t19'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux01_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t19'],2)+np.power(s5['Uy_t19'],2)+np.power(s5['Uz_t19'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t19'],2)+np.power(s6['Uy_t19'],2)+np.power(s6['Uz_t19'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t19'],2)+np.power(s7['Uy_t19'],2)+np.power(s7['Uz_t19'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t19'],2)+np.power(s8['Uy_t19'],2)+np.power(s8['Uz_t19'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t19 -------------- > Plotted......"      
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t19',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t19'],2)+np.power(s9['Uy_t19'],2)+np.power(s9['Uz_t19'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t19'],2)+np.power(s10['Uy_t19'],2)+np.power(s10['Uz_t19'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t19'],2)+np.power(s11['Uy_t19'],2)+np.power(s11['Uz_t19'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t19'],2)+np.power(s12['Uy_t19'],2)+np.power(s12['Uz_t19'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t19.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t19 -------------- > Plotted......"      
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux01_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s1.index.to_series()-s1.index[0], np.sqrt(np.power(s1['Ux_t21'],2)+np.power(s1['Uy_t21'],2)+np.power(s1['Uz_t21'],2)), label= r'$x_1 $',linewidth=line_width)
        plt.plot(s2.index.to_series()-s2.index[0], np.sqrt(np.power(s2['Ux_t21'],2)+np.power(s2['Uy_t21'],2)+np.power(s2['Uz_t21'],2)),label= r'$x_2 $',linewidth=line_width)
        plt.plot(s3.index.to_series()-s3.index[0], np.sqrt(np.power(s3['Ux_t21'],2)+np.power(s3['Uy_t21'],2)+np.power(s3['Uz_t21'],2)), label= r'$x_3 $',linewidth=line_width)
        plt.plot(s4.index.to_series()-s4.index[0], np.sqrt(np.power(s4['Ux_t21'],2)+np.power(s4['Uy_t21'],2)+np.power(s4['Uz_t21'],2)),label= r'$x_4 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux01_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux01_t21 -------------- > Plotted......"           
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux02_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s5.index.to_series()-s5.index[0], np.sqrt(np.power(s5['Ux_t21'],2)+np.power(s5['Uy_t21'],2)+np.power(s5['Uz_t21'],2)), label= r'$x_5 $',linewidth=line_width)
        plt.plot(s6.index.to_series()-s6.index[0], np.sqrt(np.power(s6['Ux_t21'],2)+np.power(s6['Uy_t21'],2)+np.power(s6['Uz_t21'],2)), label= r'$x_6 $',linewidth=line_width)
        plt.plot(s7.index.to_series()-s7.index[0], np.sqrt(np.power(s7['Ux_t21'],2)+np.power(s7['Uy_t21'],2)+np.power(s7['Uz_t21'],2)), label= r'$x_7 $',linewidth=line_width)
        plt.plot(s8.index.to_series()-s8.index[0], np.sqrt(np.power(s8['Ux_t21'],2)+np.power(s8['Uy_t21'],2)+np.power(s8['Uz_t21'],2)), label= r'$x_8 $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux02_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux02_t21 -------------- > Plotted......"    
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pylab.figure('simulated_sample_Ux03_t21',figsize=figure_size)
        pylab.clf()
        pylab.xlabel(r'${\rm X[m]}$',size=x_lable_size)
        pylab.ylabel(r'${\rm U[m/s]}$',size=y_lable_size)
        plt.plot(s9.index.to_series()-s9.index[0], np.sqrt(np.power(s9['Ux_t21'],2)+np.power(s9['Uy_t21'],2)+np.power(s9['Uz_t21'],2)), label= r'$x_9 $',linewidth=line_width)
        plt.plot(s10.index.to_series()-s10.index[0], np.sqrt(np.power(s10['Ux_t21'],2)+np.power(s10['Uy_t21'],2)+np.power(s10['Uz_t21'],2)), label= r'$x_{10} $',linewidth=line_width)
        plt.plot(s11.index.to_series()-s11.index[0], np.sqrt(np.power(s11['Ux_t21'],2)+np.power(s11['Uy_t21'],2)+np.power(s11['Uz_t21'],2)), label= r'$x_{11} $',linewidth=line_width)
        plt.plot(s12.index.to_series()-s12.index[0], np.sqrt(np.power(s12['Ux_t21'],2)+np.power(s12['Uy_t21'],2)+np.power(s12['Uz_t21'],2)), label= r'$x_{12} $',linewidth=line_width)
        pylab.gcf().autofmt_xdate()
        plt.grid(color='b', alpha=0.9, linestyle='dashed', linewidth=0.3)
        plt.legend(loc='best',prop={'size':legend_size})
        plt.gcf().autofmt_xdate()
        plt.savefig(reading_input.save_graphs +'/ABL_results/Ux03_sample_t21.tif',bbox_inches='tight')
        plt.show()
        print "sample Ux03_t21 -------------- > Plotted......"         
    print "sample Ux0i_ti -------------- > Plotted......"
    return
# ******************************************************************************************
# Set 1-------------------------------------------------------------
case = 'CFDEvap_30_01'
# ------------------------------------------------------------
point = 'X7'
ax = 'X'
#------------------------------------------------
# Samples
samples = reading_sample(case,point)
#plotting_sample_T(samples, ax)
#plotting_sample_alphat(samples, ax)
#plotting_sample_epsilon(samples, ax)
#plotting_sample_k(samples, ax)
#plotting_sample_nut(samples, ax)
#plotting_sample_p(samples, ax)
#plotting_sample_p_rgh(samples, ax)
plotting_sample_q(samples, ax)
#plotting_sample_Ux(samples, ax)
#plotting_sample_Uy(samples, ax)
#plotting_sample_Uz(samples, ax)
#plotting_sample_U(samples, ax)
#-------------------------------------------------------------
# Probes
#probes = reading_probe(case)
#plotting_probe_T(probes)
#plotting_probe_alphat(probes)
#plotting_probe_epsilon(probes)
#plotting_probe_k(probes)
#plotting_probe_nut(probes)
#plotting_probe_p(probes)
#plotting_probe_p_rgh(probes)
#plotting_probe_q(probes)
#plotting_probe_Ux(probes)
#plotting_probe_Uy(probes)
#plotting_probe_Uz(probes)
#plotting_probe_U(probes)
#Set 2 ---------------------------------------
point_z1 = 'z1'
point_z2 = 'z2'
point_z3 = 'z3'
point_z4 = 'z4'
point_z5 = 'z5'
point_z6 = 'z6'
point_z7 = 'z7'
ax = 'Z'
##samples_z1 = reading_sample(case,point_z1)
#samples_z2 = reading_sample(case,point_z2)
#samples_z3 = reading_sample(case,point_z3)
#samples_z4 = reading_sample(case,point_z4)
#samples_z5 = reading_sample(case,point_z5)
#samples_z6 = reading_sample(case,point_z6)
#samples_z7 = reading_sample(case,point_z7)
# 
#plotting_sample_T_01(samples_z2,samples_z3,samples_z4,samples_z5,samples_z6,samples_z7,ax)
#plotting_sample_epsilon_01(samples_z2,samples_z3,samples_z4,samples_z5,samples_z6,samples_z7,ax)
#plotting_sample_k_01(samples_z2,samples_z3,samples_z4,samples_z5,samples_z6,samples_z7,ax)
#plotting_sample_q_01(samples_z2,samples_z3,samples_z4,samples_z5,samples_z6,samples_z7,ax)
#plotting_sample_U_01(samples_z2,samples_z3,samples_z4,samples_z5,samples_z6,samples_z7,ax)
#Set 3 -------------------
point_x1 = 'X1'
point_x2 = 'X2'
point_x3 = 'X3'
point_x4 = 'X4'
point_x5 = 'X5'
point_x6 = 'X6'
point_x7 = 'X7'
point_x8 = 'X8'
point_x9 = 'X9'
point_x10 = 'X10'
point_x11 = 'X11'
point_x12 = 'X12'
ax = 'X'
#samples_x1 = reading_sample(case,point_x1)
#samples_x2 = reading_sample(case,point_x2)
#samples_x3 = reading_sample(case,point_x3)
#samples_x4 = reading_sample(case,point_x4)
#samples_x5 = reading_sample(case,point_x5)
#samples_x6 = reading_sample(case,point_x6)
#samples_x7 = reading_sample(case,point_x7)
#samples_x8 = reading_sample(case,point_x8)
#samples_x9 = reading_sample(case,point_x9)
#samples_x10 = reading_sample(case,point_x10)
#samples_x11 = reading_sample(case,point_x11)
#samples_x12 = reading_sample(case,point_x12)
#
#plotting_sample_T_02(samples_x1,samples_x2,samples_x3,samples_x4,samples_x5,samples_x6,samples_x7,samples_x8,samples_x9,samples_x10,samples_x11,samples_x12,ax)
#plotting_sample_k_02(samples_x1,samples_x2,samples_x3,samples_x4,samples_x5,samples_x6,samples_x7,samples_x8,samples_x9,samples_x10,samples_x11,samples_x12,ax)
#plotting_sample_epsilon_02(samples_x1,samples_x2,samples_x3,samples_x4,samples_x5,samples_x6,samples_x7,samples_x8,samples_x9,samples_x10,samples_x11,samples_x12,ax)
#plotting_sample_q_02(samples_x1,samples_x2,samples_x3,samples_x4,samples_x5,samples_x6,samples_x7,samples_x8,samples_x9,samples_x10,samples_x11,samples_x12,ax)
#plotting_sample_U_02(samples_x1,samples_x2,samples_x3,samples_x4,samples_x5,samples_x6,samples_x7,samples_x8,samples_x9,samples_x10,samples_x11,samples_x12,ax)

