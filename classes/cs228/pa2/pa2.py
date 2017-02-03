import os
import sys
import numpy as np
from scipy.misc import logsumexp
from collections import Counter
import collections
import random

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior
alpha = 0.1

def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------
class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    '''
    # likelihood p(a_i = a | c)
    self.index = A_i
    self.p = Counter({0:0, 1:0})

    pass


  def learn(self, A, C):
    '''
    TODO populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments 
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    M, N = A.shape
    self.prior = 1.*np.sum(C == 1) / M

    for c in xrange(2):
        #p_c = c*self.prior + (1 - c)*(1 - self.prior)

        mask = (C == c) & (A[:,self.index] != -1)
    
        # the + 2 and + 1 are Laplace Smoothing
        denom = 1.*np.sum(mask) + 2*alpha # np.sum(C == c)
        numer = 1.*np.sum(A[mask, self.index]) + alpha
        self.p[c] = numer/denom


class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    # probability p(c = 1)
    #self.prior = 0.5
    M, N = A_train.shape
    self.prior = 1.*np.sum(C_train == 1) / M
    
    self.cpts = []
    for i in xrange(A_train.shape[0]):
        self.cpts.append(NBCPT(i))

    self._train(A_train, C_train)

    pass


  def _train(self, A_train, C_train):
    '''
    TODO train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments 
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    nsamples, nvars = A_train.shape
    for i in xrange(nvars):
        self.cpts[i].learn(A_train, C_train)


  def classify(self, entry):
    '''
    TODO return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables 
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1

    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)

    '''
    # p(Pa(X) | X) = p(X | Pa(X)) * P(Pa(X)) / P(X)
    log_probs = collections.defaultdict(list)
    for c in xrange(2):
        p_c = c*self.prior + (1 - c)*(1 - self.prior)
        log_probs[c].append(np.log(p_c))

        for index, val in enumerate(entry):
            #import pdb
            #pdb.set_trace()
            this_p = self.cpts[index].p[c]
            if val:
                log_probs[c].append(np.log(this_p))
            else:
                # getting underflows on this
                log_probs[c].append(np.log(1 - this_p))


    # now get P(X) = sum_Pa(X) P(Pa(X)) * P(X | Pa(X))
    choice = [np.sum(log_probs[0]), np.sum(log_probs[1])]
    log_px = np.sum(choice)
    c_pred = np.argmax(choice)

    #import pdb
    #pdb.set_trace()

    return (c_pred, choice[c_pred] - log_px)
    #return (c_pred, logP_c_pred)


#--------------------------------------------------------------------------
# TANB CPT and classifier
#--------------------------------------------------------------------------
class TANBCPT(object):
  '''
  TANB CPT for a child attribute.  Each child can have one other attribute
  parent (or none in the case of the root), and the class variable as a
  parent
  '''

  def __init__(self, A_i, A_p):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
     - A_i: the index of the child variable
     - A_p: the index of its parent variable (in the Chow-Liu algorithm,
       the learned structure will have a single parent for each child)
    '''
    self.index = A_i
    self.parent = A_p

    # we need to store the probability of A_i
    # given both the class c and parent A_p
    # keys are (class, parent) tuples
    if self.parent:
        self.p = Counter({(0,0):0, (1,0):0, (0,1):0, (1,1):0})
    else:
        self.p = Counter({0:0, 1:0})


  def learn(self, A, C):
    '''
    TODO populate any instance variables specified in __init__ to learn
    the parameters for this CPT
     - A: a 2-d numpy array where each row is a sample of assignments 
     - C: a 1-d n-element numpy where the elements correspond to the class
       labels of the rows in A
    '''
    M, N = A.shape
    self.prior = 1.*np.sum(C == 1) / M

    for c in xrange(2):
        if self.parent:
            for pa in xrange(2):
                #p_c = c*self.prior + (1 - c)*(1 - self.prior)

                mask = (C == c) & (A[:,self.index] != -1) & (A[:,self.parent] == pa)
            
                # the + 2 and + 1 are Laplace Smoothing
                denom = 1.*np.sum(mask) + 2*alpha # np.sum(C == c)
                numer = 1.*np.sum(A[mask, self.index]) + alpha
                self.p[(c,pa)] = numer/denom
        else:
            mask = (C == c) & (A[:,self.index] != -1) 
            
            # the + 2 and + 1 are Laplace Smoothing
            denom = 1.*np.sum(mask) + 2*alpha # np.sum(C == c)
            numer = 1.*np.sum(A[mask, self.index]) + alpha
            self.p[c] = numer/denom



class TANBClassifier(NBClassifier):
  '''
  TANB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    _train()
        - A_train: a 2-d numpy array where each row is a sample of
          assignments 
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A

    '''
    M, N = A_train.shape
    self.prior = 1.*np.sum(C_train == 1) / M

    mst = get_mst(A_train, C_train)
    root = get_tree_root(mst)
    
    self.remaining_nodes = [x for x in xrange(N)]
    self.nodes = []
    self.nodes.append(TANBCPT(root, None))
    self.remaining_nodes.remove(root)

    edge = get_tree_edges(mst, root)
    for parent, child in edge:
        self.nodes.append(TANBCPT(child, parent))
        self.remaining_nodes.remove(child)

    for child in self.remaining_nodes:
        self.nodes.append(TANBCPT(child, None))
        self.remaining_nodes.remove(child)

    self._train(A_train, C_train)


  def _train(self, A_train, C_train):
    '''
    TODO train your TANB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of
          assignments 
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    for node in self.nodes:
        node.learn(A_train, C_train)


  def classify(self, entry):
    '''
    TODO return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables 
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1

    NOTE: this class inherits from NBClassifier and it is possible to
    write this method in NBClassifier, such that this implementation can
    be removed

    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)
    '''
        # p(Pa(X) | X) = p(X | Pa(X)) * P(Pa(X)) / P(X)
    log_probs = collections.defaultdict(list)
    for c in xrange(2):
        p_c = c*self.prior + (1 - c)*(1 - self.prior)
        log_probs[c].append(np.log(p_c))

        for node in self.nodes:
            if node.parent:
                this_p = node.p[(c,entry[node.parent])]
            else:
                this_p = node.p[c]

            if entry[node.index]:
                log_probs[c].append(np.log(this_p))
            else:
                log_probs[c].append(np.log(1 - this_p))


    # now get P(X) = sum_Pa(X) P(Pa(X)) * P(X | Pa(X))
    choice = [np.sum(log_probs[0]), np.sum(log_probs[1])]
    log_px = np.sum(choice)
    c_pred = np.argmax(choice)

    #import pdb
    #pdb.set_trace()

    return (c_pred, choice[c_pred] - log_px)


# load all data
A_base, C_base = load_vote_data()

def evaluate(classifier_cls, train_subset=False):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or TANBClassifier
  - train_subset: train the classifier on a smaller subset of the training
    data
  NOTE you do *not* need to modify this function
  
  '''
  global A_base, C_base

  A, C = A_base, C_base

  # score classifier on specified attributes, A, against provided labels,
  # C
  def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
      c_pred, _ = classifier.classify(entry)
      results.append((c_pred == c))
      pp.append(_)
    #print 'logprobs', np.array(pp)
    return results
  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  step = M / 10
  for holdout_round, i in enumerate(xrange(0, M, step)):
    A_train = np.vstack([A[0:i,:], A[i+step:,:]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i:i+step,:]
    C_test = C[i:i+step]
    if train_subset:
      A_train = A_train[:16,:]
      C_train = C_train[:16]

    # train the classifiers
    classifier = classifier_cls(A_train, C_train)
  
    train_results = get_classification_results(classifier, A_train, C_train)
    #print '  train correct {}/{}'.format(np.sum(nb_results), A_train.shape[0])
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)

  return 1.*tot_correct/tot_test, tot_test

def evaluate_incomplete_entry(classifier_cls):

  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)

  print '  P(C={}|A_observed) = {:2.4f}'.format(c_pred, np.exp(logP_c_pred))

  return

def main():
  '''
  TODO modify or add calls to evaluate() to evaluate your implemented
  classifiers
  '''
  print 'Naive Bayes'
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

  print 'TANB Classifier'
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
  print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

  print 'Naive Bayes Classifier on missing data'
  evaluate_incomplete_entry(NBClassifier)

  print 'TANB Classifier on missing data'
  evaluate_incomplete_entry(TANBClassifier)

if __name__ == '__main__':
  main()

