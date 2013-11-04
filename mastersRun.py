# Master's Thesis iPython Notebook scripts to run

import numpy as np
from pylab import *
from scipy import *
from matplotlib import *
import datetime
import pickle

from mastersFunctions import *
from emailing import *


try:
    M     = 70
    nBins = 500
    adapt = linspace(0,25,100)
    
    steady_state_mem  = []
    steady_state_pred = []
    steady_state_max  = []
    sum_mem           = []
    sum_pred          = []
    sum_max           = []
    i_mem_fnA         = np.zeros((3,999))
    i_pred_fnA        = np.zeros((3,999))
    i_max_fnA         = np.zeros((3,999))
    
    for a in adapt:
        # run simulation
        V,w,spikes,sptimes,T,stimulus = ensemble(a,int(M),'brownian',1000)
        
        V        = np.asarray(V)
        stimulus = np.asarray(stimulus)
        
        # compute i_mem, i_pred, i_max
        i_mem   = []
        i_pred  = []
        i_max   = []

        minV = min(flatten(V))
        maxV = max(flatten(V))
        minC = min(flatten(stimulus))
        maxC = max(flatten(stimulus))

        for i in xrange(shape(V)[1]-1):
            H, I = mutiN(V[:,i],stimulus[:,i],nBins,minV,maxV,minC,maxC)
            i_mem.append(I)
            H, I = mutiN(V[:,i],stimulus[:,i+1],nBins,minV,maxV,minC,maxC)
            i_pred.append(I)
            H, I = mutiN(stimulus[:,i],stimulus[:,i+1],nBins,minC,maxC,minC,maxC)
            i_max.append(I)
    
        steady_state_mem.append(i_mem[-1])
        steady_state_pred.append(i_pred[-1])
        steady_state_max.append(i_max[-1])
        
        sum_mem.append(sum(i_mem))
        sum_pred.append(sum(i_pred))
        sum_max.append(sum(i_max))
        
        if a < 0.1:
            i_mem_fnA[0,:]  = asarray(i_mem)
            i_pred_fnA[0,:] = asarray(i_pred)
            i_max_fnA[0,:]  = asarray(i_max)
        elif abs(a - 6) < 0.1:
            i_mem_fnA[1,:]  = asarray(i_mem)
            i_pred_fnA[1,:] = asarray(i_pred)
            i_max_fnA[1,:]  = asarray(i_max)
        elif abs(a - 25) < 0.1:
            i_mem_fnA[2,:]  = asarray(i_mem)
            i_pred_fnA[2,:] = asarray(i_pred)
            i_max_fnA[2,:]  = asarray(i_max)
    
    

    with open('masters_data_steadyState_' + str(datetime.date.today()) + '.pik', 'wb') as f:
        pickle.dump([steady_state_mem, steady_state_pred, steady_state_max])

    with open('masters_data_sum_' + str(datetime.date.today()) + '.pik', 'wb') as f:
        pickle.dump([sum_mem, sum_pred, sum_max])

    with open('masters_data_examples_' + str(datetime.date.today()) + '.pik', 'wb') as f:
        pickle.dump([i_mem_fnA, i_pred_fnA, i_max_fnA])



    emailWhenDone()
    
except:
    import traceback
    import smtplib
    
    sysexecinfo = sys.exc_info()
    
    emailWhenError(sysexecinfo)




