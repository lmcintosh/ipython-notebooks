###############################################################################
# Finishes PA 3
# author: Billy Jun, Xiaocheng Li
# date: Jan 31, 2016
###############################################################################

## Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cluster_graph import *
from factors import *

def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices
  
    return values:
    G: generator matrix
    H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H

def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)  
  
    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, p):
    '''
    :param y - codeword with 2N entries
    :param p channel noise probability
  
    return corrupt message yhat  
    yhat_i is obtained by flipping y_i with probability p 
    '''
    ###############################################################################
    # TODO: Your code here! 

    
    
    ###############################################################################
    return yhat


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructClusterGraph(yhat, H, p):
    '''
    :param - yhat: observed codeword
    :param - H parity check matrix
    :param - p channel noise probability

    return G clusterGraph
   
    You should consider two kinds of factors:
    - M unary factors 
    - N each parity check factors
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = ClusterGraph(M)
    domain = [0, 1]
    G.nbr = [[] for _ in range(M+N)]
    G.sepset = [[None for _ in range(M+N)] for _ in range(M+N)]
    ##############################################################
    # To do: your code starts here
 


    ##############################################################
    return G

def do_part_a():
    yhat = np.array([[1, 1, 1, 1, 1]]).reshape(5,1)
    H = np.array([ \
        [0, 1, 1, 0, 1], \
        [0, 1, 0, 1, 1], \
        [1, 1, 0, 1, 0], \
        [1, 0, 1, 1, 0], \
        [1, 0, 1, 0, 1]])
    p = 0.95
    G = constructClusterGraph(yhat, H, p)
    ##############################################################
    # To do: your code starts here 
    # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
    # Report their weights respectively.
    ytest1 = []
    ytest2 = []
    ytest3 = []
    ##############################################################
    print(
        G.evaluateWeight(ytest1), \
        G.evaluateWeight(ytest2), \
        G.evaluateWeight(ytest3))

def do_part_c():
    '''
    In part b, we provide you an all-zero initialization of message x, you should
    apply noise on y to get yhat, znd then do loopy BP to obatin the
    marginal probabilities of the unobserved y_i's.
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    p = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    ##############################################################
    # To do: your code starts here



    ##############################################################

def do_part_de(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    ##############################################################
    # To do: your code starts here



    ##############################################################

def do_part_fg(error):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the previous parts.



    ################################################################

print('Doing part (a): Should see 0.0, 0.0, >0.0')
#do_part_a()
print('Doing part (c)')
#do_part_c()
print('Doing part (d)')
#do_part_de(10, 0.06)
print('Doing part (e)')
#do_part_de(10, 0.08)
#do_part_de(10, 0.10)
print('Doing part (f)')
#do_part_fg(0.06)
print('Doing part (g)')
#do_part_fg(0.10)

