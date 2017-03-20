# Learning: GMM and EM
# Authors: Gunaa A V, Isaac C, Volodymyr K, Haque I, Aditya G
# Last updated: March 3, 2017

import numpy as np
from pprint import pprint
import copy
import math
import time
import matplotlib.pyplot as plt

LABELED_FILE = "surveylabeled.dat"
UNLABELED_FILE = "surveyunlabeled.dat"


#===============================================================================
# General helper functions

def colorprint(message, color="rand"):
    """Prints your message in pretty colors! 

    So far, only the colors below are available.
    """
    if color == 'none': print message; return
    if color == 'demo':
        for i in range(99):
            print '%i-'%i + '\033[%sm'%i + message + '\033[0m\t',
    print '\033[%sm'%{
        'neutral' : 99,
        'flashing' : 5,
        'underline' : 4,
        'magenta_highlight' : 45,
        'red_highlight' : 41,
        'pink' : 35,
        'yellow' : 93,   
        'teal' : 96,     
        'rand' : np.random.randint(1,99),
        'green?' : 92,
        'red' : 91,
        'bold' : 1
    }.get(color, 1)  + message + '\033[0m'

def read_labeled_matrix(filename):
    """Read and parse the labeled dataset.

    Output:
        Xij: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        Zij: dictionary of party choices.
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a float.
        N, M: Counts of precincts and voters.
    """
    Zij = {} 
    Xij = {}
    M = 0.0
    N = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, Z, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i>N: N = i
            if j>M: M = j

            Zij[i-1, j-1] = float(Z)
            Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
    return Xij, Zij, N, M

def read_unlabeled_matrix(filename):
    """Read and parse the unlabeled dataset.
    
    Output:
        Xij: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters.
    """
    Xij = {}
    M = 0.0
    N = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i>N: N = i
            if j>M: M = j

            Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
    return Xij, N, M


#===============================================================================
# Functions that define the probability distribution
#
# There are helper functions that you may find useful for solving the problem.
# You do not need to use them, but we have found them to be helpful.
# Not all of them are implemented. We give a few examples, and you should 
# implement the rest. We suggest you first start by implementing the rest of the
# homework and fill-in the missing functions here when you realize that you need
# them.

def p_yi(y_i, phi):
    """Probability of y_i.

    Bernouilli distribution with parameter phi.
    """
    return (phi**y_i) * ((1-phi)**(1-y_i))

def p_zij(z_ij, pi):
    """Probability of z_ij.

    Bernouilli distribution with parameter pi.
    """
    return (pi**z_ij) * ((1-pi)**(1-z_ij))

def p_zij_given_yi(z_ij, y_i, lambd):
    """Probability of z_ij given yi.

    Bernouilli distribution with parameter lambd that targets
    the variable (z_ij == y_i).
    """
    if z_ij == y_i:
        return lambd
    return 1-lambd

def z_marginal(z_ij, lambd, phi):
    """Marginal probability of z_ij with yi marginalized out."""
    return p_zij_given_yi(z_ij, 1, lambd) * p_yi(1, phi) \
         + p_zij_given_yi(z_ij, 0, lambd) * p_yi(0, phi)

def p_xij_given_zij(x_ij, mu_zij, sigma_zij):
    """Probability of x_ij.

    Given by multivariate normal distribution with params mu_zij and sigma_zij.
    
    Input:
        x_ij: (1,2) array of continuous variables
        mu_zij: (1,2) array representing the mean of class z_ij
        sigma_zij: (2,2) array representing the covariance matrix of class z_ij

    All arrays must be instances of numpy.matrix class.
    """
    assert isinstance(x_ij, np.matrix)
    k = x_ij.shape[1]; assert(k==2)

    det_sigma_zij = sigma_zij[0, 0]*sigma_zij[1, 1] - sigma_zij[1, 0]*sigma_zij[0, 1]
    assert det_sigma_zij > 0

    sigma_zij_inv = -copy.copy(sigma_zij); sigma_zij_inv[0, 0] = sigma_zij[1, 1]; sigma_zij_inv[1, 1] = sigma_zij[0, 0]
    sigma_zij_inv /= det_sigma_zij

    # print "better be identity matrix:\n", sigma_zij.dot(sigma_zij_inv)

    multiplicand =  (((2*math.pi)**k)*det_sigma_zij)**(-0.5)
    exponentiand = -.5 * (x_ij-mu_zij).dot(sigma_zij_inv).dot((x_ij-mu_zij).T)
    exponentiand = exponentiand[0,0]
    return multiplicand * np.exp(exponentiand)

def p_zij_given_xij_unnorm(z_ij, x_ij, lambd, phi, mu_zij, sigma_zij):
    """Unnormalized posterior probability of z_ij given x_ij."""

    # -------------------------------------------------------------------------
    # (Optional): Put your code here
    
    pass

    # END_YOUR_CODE
    

def p_xij_given_yi(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, lambd):
    """Probability of x_ij given y_i.
    
    To compute this, marginalize (i.e. sum out) over z_ij.
    """

    # -------------------------------------------------------------------------
    # TODO (Optional): Put your code here
    
    pass

    # END_YOUR_CODE
    

def p_yi_given_xij_unnorm(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, phi, lambd):
    """Unnormalized posterior probability of y_ij given x_ij.
    
    Hint: use Bayes' rule!
    """

    # -------------------------------------------------------------------------
    # TODO (Optional): Put your code here
    
    pass

    # END_YOUR_CODE

def MLE_Estimation(Xij=None, Zij=None):
    """Perform MLE estimation of Model A parameters.

    Output:
        pi: (float), estimate of party proportions
        mean0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
        mean1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
        sigma0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
        sigma1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
    """
    
    if (not Xij) or (not Zij):
        # in this case, N = 5, M = 20
        Xij, Zij, N, M = read_labeled_matrix(LABELED_FILE)
    else:
        N, M = (50, 20)

    pi = 0.0
    # -------------------------------------------------------------------------
    # MLE pi is just the empirical ratio of party proportions
    # i.e. fraction of Z's that are 1.0
    pi = sum([Zij[k] for k in Zij.keys()])/float(N*M)
    # END_YOUR_CODE
    

    mean0 = np.matrix([0.0, 0.0])
    mean1 = np.matrix([0.0, 0.0])
    # -------------------------------------------------------------------------
    # refactoring code to handle non-integer Zij's
    #mean0 = np.mean(np.stack([Xij[k] for k in Xij.keys() if Zij[k] < 1]), axis=0)
    #mean1 = np.mean(np.stack([Xij[k] for k in Xij.keys() if Zij[k] > 0]), axis=0)
    z_sum = float(sum([Zij[k] for k in Zij.keys()]))
    z_complement = N*M - z_sum
    mean0 = (1./z_complement)*np.sum(np.stack([(1. - Zij[k]) * Xij[k] for k in Xij.keys()]), axis=0)
    mean1 = (1./z_sum)*np.sum(np.stack([Zij[k] * Xij[k] for k in Xij.keys()]), axis=0)
    # END_YOUR_CODE


    sigma0 = np.matrix([[0.0,0.0],[0.0,0.0]])
    sigma1 = np.matrix([[0.0,0.0],[0.0,0.0]])
    # -------------------------------------------------------------------------
    class_zeros = np.stack([Xij[k] for k in sorted(Xij.keys())]) - mean0 
    Zij_zeros = np.array([(1. - Zij[k]) for k in sorted(Xij.keys())]).reshape((N*M,1))
    class_ones = np.stack([Xij[k] for k in sorted(Xij.keys())]) - mean1 
    Zij_ones = np.array([Zij[k] for k in sorted(Xij.keys())]).reshape((N*M,1))

    sigma0 = (1./z_complement) * np.dot(class_zeros.T, np.multiply(class_zeros, Zij_zeros))
    sigma1 = (1./z_sum) * np.dot(class_ones.T, np.multiply(class_ones, Zij_ones))
    # END_YOUR_CODE

    return pi, mean0, mean1, sigma0, sigma1


def perform_em_modelA(X, N, M, init_params, max_iters=50, eps=1e-2):
    """Estimate Model A paramters using EM

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        init_params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 

    Output:
        params: parameters of the trained model encoded as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
        log_likelihood: array of log-likelihood values across iterations
    """

    def compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi):
        """Compute the log-likelihood of the data given our parameters

        Input:
            pi: (float), estimate of party proportions
            mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
            mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
            sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
            sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1

        Output:
            ll: (float), value of the log-likelihood
        """
        ll = 0.0
        # -------------------------------------------------------------------------
        for i in range(N):
            for j in range(M):
                zero_term = (1. - pi) * p_xij_given_zij(X[(i,j)], mu_0, sigma_0)
                one_term = pi * p_xij_given_zij(X[(i,j)], mu_1, sigma_1)
                ll += np.log(zero_term + one_term)

        # END_YOUR_CODE
        return ll
    
    # unpack the parameters

    mu_0 = init_params['mu_0']
    mu_1 = init_params['mu_1']
    sigma_0 = init_params['sigma_0']
    sigma_1 = init_params['sigma_1']
    pi = init_params['pi']

    # set up list that will hold log-likelihood over time
    log_likelihood = [compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi)]

    for iter in xrange(max_iters):
        # -------------------------------------------------------------------------
        posterior = {}
        for i in range(N):
            for j in range(M):
                # p_xij_given_zij is just a multivariate normal
                # p(z=0) * p(x | z=0)
                zero_term = (1. - pi) * p_xij_given_zij(X[(i,j)], mu_0, sigma_0)
                # p(z=1) * p(x | z=1)
                one_term = pi * p_xij_given_zij(X[(i,j)], mu_1, sigma_1)
                # p(z = 1 | x) = p(z=1) * p(x | z=1) / sum_z p(z) p(x | z)
                posterior[(i,j)] = one_term / (zero_term + one_term)
        # END_YOUR_CODE

        pi = 0.0
        mu_0 = np.matrix([0.0, 0.0])
        mu_1 = np.matrix([0.0, 0.0])
        sigma_0 = np.matrix([[0.0,0.0],[0.0,0.0]])
        sigma_1 = np.matrix([[0.0,0.0],[0.0,0.0]])

        # -------------------------------------------------------------------------
        # Code for the M step
        # You should fill the values of pi, mu_0, mu_1, sigma_0, sigma_1
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation(X, posterior)
        # END_YOUR_CODE
        
        # Check for convergence
        this_ll = compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, pi)
        log_likelihood.append(this_ll)
        if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
            break

    # pack the parameters and return
    params = {}
    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['pi'] = pi

    return params, log_likelihood

def MLE_of_phi_and_lamdba(X=None, Y=None, Z=None):
    """Perform MLE estimation of Model B parameters.

    Assumes that Y variables have been estimated using heuristic proposed in
    the question.

    Output:
        MLE_phi: estimate of phi
        MLE_lambda: estimate of lambda
    """
    if (not X) or (not Z):
        X, Z, N, M = read_labeled_matrix(LABELED_FILE)
        assert(len(Z.items()) == M*N)
    else:
        N = 50
        M = 20

    MLE_phi, MLE_lambda = 0.0, 0.0


    # -------------------------------------------------------------------------
    # Code to compute MLE_phi, MLE_lambda 
    # first set yi to the consensus of zij's
    if not Y:
        Y = {}
        for i in range(N):
            this_precinct = sum([Z[(i,j)] for j in range(M)])
            if this_precinct > 0.5*M:
                Y[i] = 1
            else:
                Y[i] = 0

    # now estimate phi and lambda
    MLE_phi = (1.0/N) * sum([Y[k] for k in Y.keys()])
    equalities = 0
    inequalities = 0
    for i in range(N):
        for j in range(M):
            equalities += Z[(i,j)]*Y[i] + (1 - Z[(i,j)])*(1 - Y[i])
            inequalities += Z[(i,j)]*(1 - Y[i]) + (1 - Z[(i,j)])*Y[i]
    MLE_lambda = float(equalities) / float(equalities + inequalities)
    # END_YOUR_CODE

    return MLE_phi, MLE_lambda

def estimate_leanings_of_precincts(X, N, M, params=None):
    """Estimate the leanings y_i given data X.

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        params: parameters of the model given as a dictionary
            Dictionary should contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']

    Output:
        Summary: length-N list summarizing the leanings
            Format is: [(i, prob_i, y_i) for i in range(N)]
    """
    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']


    posterior_y = [None for i in range(N)] 
    # -------------------------------------------------------------------------
    # Code to compute posterior_y
    log_p_y1 = np.log(MLE_phi)
    log_p_y0 = np.log(1.0 - MLE_phi)
    for i in range(N):
        posterior_y1 = log_p_y1
        posterior_y0 = log_p_y0
        for j in range(M):
            p_x_z1 = p_xij_given_zij(X[(i,j)], mu_1, sigma_1)
            p_x_z0 = p_xij_given_zij(X[(i,j)], mu_0, sigma_0)
            posterior_y1 += np.log(MLE_lambda*p_x_z1 + (1. - MLE_lambda)*p_x_z0)
            posterior_y0 += np.log((1.-MLE_lambda)*p_x_z1 + MLE_lambda*p_x_z0)

        normalization = np.exp(posterior_y1) + np.exp(posterior_y0)
        posterior_y[i] = np.exp(posterior_y1)/normalization
    # END_YOUR_CODE

    summary = [(i, p, 1 if p>=.5 else 0) for i, p in enumerate(posterior_y)]
    return summary

def plot_individual_inclinations(X, N, M, params=None):
    """Generate 2d plot of individual statistics in each class.

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        params: parameters of the model given as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
    """

    if params == None:
        pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()    
        MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    else:
        pi = params['pi']
        mu_0 = params['mu_0']
        mu_1 = params['mu_1']
        sigma_0 = params['sigma_0']
        sigma_1 = params['sigma_1']
        MLE_phi = params['phi']
        MLE_lambda = params['lambda']

    domain0 = []
    range0 = []
    domain1 = []
    range1 = []

    posterior_y = estimate_leanings_of_precincts(X, N, M, {'pi': pi,
                                                           'mu_0': mu_0,
                                                           'mu_1': mu_1,
                                                           'sigma_0': sigma_0,
                                                           'sigma_1': sigma_1,
                                                           'phi': MLE_phi,
                                                           'lambda': MLE_lambda})
    for (i, j), x_ij in X.items():
        posterior_z = [0.0, 0.0]

        # -------------------------------------------------------------------------
        # Code to compute posterior_z
        p_x_z0 = p_xij_given_zij(x_ij, mu_0, sigma_0)
        p_x_z1 = p_xij_given_zij(x_ij, mu_1, sigma_1)

        p_z0 = (posterior_y[i][1]*(1. - MLE_lambda) + (1. - posterior_y[i][1])*MLE_lambda)*p_x_z0
        p_z1 = (posterior_y[i][1]*MLE_lambda + (1. - posterior_y[i][1])*(1. - MLE_lambda))*p_x_z1

        posterior_z[0] = p_z0/(p_z0 + p_z1)
        posterior_z[1] = p_z1/(p_z0 + p_z1)
        # END_YOUR_CODE

        # if z = 1 is more likely
        if posterior_z[1] >= posterior_z[0]:
            domain0.append(x_ij[0, 0])
            range0.append(x_ij[0, 1])
        # otherwise z = 0 is more likely
        else:
            domain1.append(x_ij[0, 0])
            range1.append(x_ij[0, 1]) 

    plt.plot(domain1, range1, 'r+')          
    plt.plot(domain0, range0, 'b+')
    p1,  = plt.plot(mu_0[0,0], mu_0[0,1], 'kd')
    p2,  = plt.plot(mu_1[0,0], mu_1[0,1], 'kd')
    plt.show()  


def perform_em(X, N, M, init_params, max_iters=50, eps=1e-2):
    """Estimate Model B paramters using EM

    Input:
        X: dictionary of measured statistics
            Dictionary is indexed by tuples (i,j).
            The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
        N, M: Counts of precincts and voters
        init_params: parameters of the model given as a dictionary
            Dictionary shoudl contain: params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']

    Output:
        params: parameters of the trained model encoded as a dictionary
            Dictionary shoudl contain params['pi'], params['mu_0'], 
            params['mu_1'], params['sigma_0'], params['sigma_1'], 
            params['phi'], params['lambda']
        log_likelihood: array of log-likelihood values across iterations
    """

    def compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd):
        """Compute the log-likelihood of the data given our parameters

        Input:
            mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
            mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
            sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
            sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
            phi: hyperparameter for princinct preferences
            lambd: hyperparameter for princinct preferences

        Output:
            ll: (float), value of the log-likelihood
        """
        ll = 0.0
        
        # -------------------------------------------------------------------------
        # Code to compute ll
        for i in range(N):
            for j in range(M):
                alpha_1 = (phi*lambd + (1 - phi)*(1 - lambd))*p_xij_given_zij(X[(i,j)], mu_1, sigma_1)
                alpha_0 = ((1 - phi)*lambd + phi*(1 - lambd))*p_xij_given_zij(X[(i,j)], mu_0, sigma_0)
                ll += np.log(alpha_1 + alpha_0)
        # END_YOUR_CODE
            
        return ll

    mu_0 = init_params['mu_0']
    mu_1 = init_params['mu_1']
    sigma_0 = init_params['sigma_0']
    sigma_1 = init_params['sigma_1']
    phi = init_params['phi']
    lambd = init_params['lambda']

    log_likelihood = [compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd)]
    
    for iter in xrange(max_iters):
        
        # -------------------------------------------------------------------------
        # Code for the E step
        posterior_y_x = {}
        log_p_y1 = np.log(phi)
        log_p_y0 = np.log(1.0 - phi)
        for i in range(N):
            posterior_y1 = log_p_y1
            posterior_y0 = log_p_y0
            for j in range(M):
                p_x_z1 = p_xij_given_zij(X[(i,j)], mu_1, sigma_1)
                p_x_z0 = p_xij_given_zij(X[(i,j)], mu_0, sigma_0)
                posterior_y1 += np.log(lambd*p_x_z1 + (1. - lambd)*p_x_z0)
                posterior_y0 += np.log((1.-lambd)*p_x_z1 + lambd*p_x_z0)

            normalization = np.exp(posterior_y1) + np.exp(posterior_y0)
            posterior_y_x[i] = np.exp(posterior_y1)/normalization


        posterior_z_x = {}
        for i in range(N):
            for j in range(M):
                p_x_z1 = p_xij_given_zij(X[(i,j)], mu_1, sigma_1)
                p_x_z0 = p_xij_given_zij(X[(i,j)], mu_0, sigma_0)
                p_z1 = (posterior_y_x[i]*lambd + (1 - posterior_y_x[i])*(1 - lambd))*p_x_z1
                p_z0 = (posterior_y_x[i]*(1 - lambd) + (1 - posterior_y_x[i])*lambd)*p_x_z0
                posterior_z_x[(i,j)] = p_z1 / (p_z1 + p_z0)

        # END_YOUR_CODE

        phi, lambd = 0.0, 0.0
        mu_0 = np.matrix([0.0, 0.0])
        mu_1 = np.matrix([0.0, 0.0])
        sigma_0 = np.matrix([[0.0,0.0],[0.0,0.0]])
        sigma_1 = np.matrix([[0.0,0.0],[0.0,0.0]])


        # -------------------------------------------------------------------------
        # Code for the M step
        # You need to compute the above variables
        _, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation(X, posterior_z_x)
        phi, lambd = MLE_of_phi_and_lamdba(X=X, Y=posterior_y_x, Z=posterior_z_x)

        # END_YOUR_CODE

        # Check for convergence
        this_ll = compute_log_likelihood(mu_0, mu_1, sigma_0, sigma_1, phi, lambd)
        log_likelihood.append(this_ll)
        if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
            break

    # pack the parameters and return
    params = {}
    params['pi'] = init_params['pi']
    params['mu_0'] = mu_0
    params['mu_1'] = mu_1
    params['sigma_0'] = sigma_0
    params['sigma_1'] = sigma_1
    params['lambda'] = lambd
    params['phi'] = phi

    return params, log_likelihood

#===============================================================================
# This runs the functions that you have defined to produce the answers to the
# assignment problems

