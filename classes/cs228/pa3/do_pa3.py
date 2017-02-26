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
    random_matrix = np.random.rand(*y.shape)
    yhat = np.where(random_matrix < p, 1 - y, y)
    
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
    for row in range(N):
        # indicate which variables have connections in the adjacency matrix H
        this_scope = [i for i,v in enumerate(H[row]) if v > 0]
        this_card = [2]*len(this_scope)

        # gather all of the 2^n possible assignments in order
        assigns = indices_to_assignment(np.arange(2**len(this_scope)), this_card)
        # assign 0 probability to assignments with mod % 2 = 1
        this_val = 1.0 - (np.sum(assigns, axis=1) % 2)

        f = Factor(scope=this_scope, card=this_card, val=this_val, name="parity")
        G.factor.append(f)

    # unary factors
    for col in range(M):
        # p(yhat_col = 0 | yhat) and p(yhat_col = 1 | yhat), respectively
        if yhat[col] == 1:
            this_val = np.array([p, 1.0-p])
        else:
            this_val = np.array([1.0-p, p])
        f = Factor(scope=[col], card=[2], val=this_val, name="unary")
        G.factor.append(f)



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
    ytest1 = [1, 0, 1, 0, 1]
    ytest2 = [1, 1, 1, 1, 1]
    ytest3 = [0, 0, 0, 0, 0]
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

    # apply noise on y
    yhat = applyChannelNoise(y, p)

    # initialize graph
    Graph = constructClusterGraph(yhat, H, p)

    # initialize varToCliques
    neighbor_vars = [f.scope for f in Graph.factor]
    for var_i in range(len(Graph.varToCliques)):
        for fac_j,neighbors in enumerate(neighbor_vars):
            if var_i in neighbors:
                Graph.varToCliques[var_i].append(fac_j)

    Graph.sepset = [[[] for j in xrange(len(Graph.factor))]
                    for i in xrange(len(Graph.factor))]

    # initialize nbr and sepset
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i][j]:
                Graph.nbr[i].append(H.shape[0] + j)
                Graph.nbr[H.shape[0] + j].append(i)
                Graph.sepset[i][H.shape[0] + j].append(j)
                Graph.sepset[H.shape[0] + j][i].append(j)

    # initialize messages
    Graph.messages = [[None for dst in range(len(Graph.factor))]
                        for src in range(len(Graph.factor))]
    for src in range(len(Graph.factor)):
        for dst in Graph.nbr[src]:
            this_scope = Graph.sepset[src][dst]
            this_card = [2]*len(this_scope)
            Graph.messages[src][dst] = Factor(scope=this_scope,
                                            card=this_card,
                                            val=np.tile(1.0, this_card))
    


    iterations = 50
    Graph.runParallelLoopyBP(iterations)

    # collect probabilities that bits = 1
    marginals = [Graph.estimateMarginalProbability(i)[1] for i in range(len(yhat))]

    #import pdb
    #pdb.set_trace()

    # plot it
    plt.scatter(range(len(yhat)), marginals, s=10, color='k')



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

    #### In ipython notebook!!! ####



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
do_part_a()
print('Doing part (c)')
#do_part_c()
#print('Doing part (d)')
#do_part_de(10, 0.06)
#print('Doing part (e)')
#do_part_de(10, 0.08)
#do_part_de(10, 0.10)
#print('Doing part (f)')
#do_part_fg(0.06)
#print('Doing part (g)')
#do_part_fg(0.10)


