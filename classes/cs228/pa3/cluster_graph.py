###############################################################################
# cluster graph data structure implementation (similar as the CliqueTree
# implementation in PA2)
# author: Billy Jun, Xiaocheng Li
# date: Jan 31st, 2016
###############################################################################

from factors import *
import numpy as np
from tqdm import tqdm

class ClusterGraph:
    def __init__(self, numVar=0):
        '''
        var - list: index/names of variables
        domain - list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]
        varToCliques - list of lists: the i-th element is a list with the indices 
                     of cliques/factors that contain the i-th variable
        nbr - list of lists: each j-th entry contains the Y variables in the scope of the j-th factor 
                (for parity and unary factors)
        factor: a list of Factors
        sepset: two dimensional array, sepset[i][j] is a list of variables shared by 
                factor[i] and factor[j]
        messages: a dictionary to store the messages, keys are (src, dst) pairs, values are 
                the Factors of sepset[src][dst]. Here src and dst are the indices for factors.
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [None for _ in range(numVar)]
        self.varToCliques = [[] for _ in range(numVar)]
        self.nbr = []
        self.factor = []
        self.sepset = []
        self.messages = []
    
    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigments
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factor:
            output *= f.val[assignment_to_indices([a[f.scope]], f.card)]
        return output[0]
    
    def getInMessage(self, src, dst):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        return: Factor with var set as sepset[src][dst]
        
        In this function, the message will be initialized as an all-one vector if 
        it is not computed and used before. 
        '''
        if (src, dst) not in self.messages:
            inMsg = Factor()
            inMsg.scope = self.sepset[src][dst]
            inMsg.card = [len(self.domain[s]) for s in inMsg.scope]
            inMsg.val = np.ones(np.prod(inMsg.card))
            self.messages[(src, dst)] = inMsg
        return self.messages[(src, dst)]

    def runParallelLoopyBP(self, iterations): 
        '''
        param - iterations: the number of iterations you do loopy BP
          
        In this method, you need to implement the loopy BP algorithm. The only values 
        you should update in this function is self.messages. 
        
        Warning: Don't forget to normalize the message at each time. You may find the normalize
        method in Factor useful.
        '''
        #import pdb
        #pdb.set_trace()
        ordering = []
        for s,fs in enumerate(self.factor):
            for i in self.nbr[s]:
                ordering.append((s, i))

        #for iter in tqdm(range(iterations)):
        for iter in range(iterations):
        ###############################################################################
        # To do: your code here
            # go through each factor updating the fac to var messages via the var to fac messages
            np.random.shuffle(ordering)

            for (s, i) in ordering:
                prod = self.factor[s]
                #key = 'var %d, fac %d' %(i, s)
                #key = 'fac %d, var %d' %(s, i)
                for neighbor in self.nbr[s]:
                    if i != neighbor:
                        # multiply var to fac messages
                        #prod_key = 'var %d, fac %d' %(neighbor, s)
                        prod = prod.multiply(self.messages[neighbor][s])
                        prod = prod.normalize()
                prod = prod.marginalize_all_but(self.sepset[s][i])
                self.messages[s][i] = prod


        ###############################################################################
        

    def estimateMarginalProbability(self, var):
        '''
        param - var: a single variable index
        return: the marginal probability of the var
        
        example: 
        >>> cluster_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]
    
        Since in this assignment, we only care about the marginal 
        probability of a single variable, you only need to implement the marginal 
        query of a single variable.     
        '''
        ###############################################################################
        # To do: your code here  
        # the last factor in varToCliques is the unary factor
        # let's start with that for our marginalization
        last_factor = self.varToCliques[var][-1]
        marginal_p = self.factor[last_factor]
        # for variables connected to this factor
        #import pdb
        #pdb.set_trace()

        for neighbor in self.nbr[last_factor]:
            #key = 'fac %d, var %d' %(f, var) 
            marginal_p = marginal_p.multiply(self.messages[neighbor][last_factor])
            marginal_p = marginal_p.normalize()
        marginal_p = marginal_p.marginalize_all_but([var])

        return marginal_p.val

        ###############################################################################
    

    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP 
        assignments for the variables. You may utilize the method
        estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        >>> cluster_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''
        
        output = np.zeros(len(self.var))
        ###############################################################################
        # To do: your code here  
        

        
        
        ###############################################################################  
        return output
