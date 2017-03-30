import numpy as np

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Matrices and vectors are numpy arrays!

############################################################################
###################                          ###############################
################### Make Bistochastic Matrix ###############################
###################                          ###############################

### The following will test convergence! 
### however I do not know what kind of test it is! 
### I am just writting (trying to write) Matt's code here!
def convergedTestBio(matrix):
    localMatrix = np.copy(matrix).astype(float);
    e1 = sum(abs(np.sum(localMatrix , axis = 0) - 1));
    e2 = sum(abs(np.sum(localMatrix , axis = 1) - 1));
    return (e1 + e2) > 0.000001
    
#############################################################################
################### Here we will do what OneStep does                 #######
################### in Matt's code inside MakeBistochastic function.  #######
def OneStepBio(matrix):
    # copy the input so that the original input is not changed.
    localMatrix = np.copy(matrix).astype(float);
    
    # Divide each row by sum of the entries in the given row.
    localMatrix = np.dot(np.diag(1/np.sum(localMatrix, axis=1)), localMatrix);
    
    # Divide each column by sum of the elements in the given column.
    localMatrix = np.dot(localMatrix, np.diag(1/np.sum(localMatrix, axis=0)));
    return localMatrix
    

def MakeBistochastic(matrix):
    localMatrix = np.copy(matrix).astype(float);
    while (convergedTestBio(localMatrix)):
        localMatrix = OneStepBio(localMatrix);
    return localMatrix

##########################################################################
###################                 ######################################
################### Make Big Matrix ######################################
###################                 ######################################
############ p communities of n individuals each #########################

def MakeUpper(n):
    noElements = n * (n+1) / 2;
    size = 1 + (-1 + np.sqrt(1 + 8 * noElements)) / 2;
    upper = np.zeros([size, size]);
    upper[np.tril_indices(size, -1)] = np.arange(1, noElements+1)
    return upper        
    
def MakeBigMatrix(n, p, ubound):
    # This function makes a adjacency matrix whose entries
    # will be probabilities of interactions with p communites of size n.

    # probability of interactions within each community are equal, 
    # but between communities are different.
    
    # This matrix will turn into a bio-stochastic later.
    BigMatrix = np.zeros((n*p , n*p));
    d = (1. / (n-1)) * np.ones((n,n)) - np.diag(np.ones(n) / (n-1));
    for rowCount in range(p):
        for colCount in range(rowCount, p):
            rowStart = rowCount * n;
            rowEnd   = rowCount * n + n ;
            colStart = colCount * n;
            colEnd   = colCount * n + n;
            if rowCount == colCount:
                # Here we take care of blocks on diagonal of 
                # bigMatrix where corresponds to probabilities within
                # each communication
                BigMatrix[ rowStart : rowEnd , colStart : colEnd ] = d
            else:
                # Here we take care of interactions between different
                # communities. The counting purposes are done for
                # lower part of bigMatrix.
                BigMatrix[rowStart:rowEnd, colStart:colEnd] = np.random.uniform(low = 0,high = ubound, size = (n, n));
                BigMatrix[colStart:colEnd, rowStart:rowEnd] = BigMatrix[rowStart:rowEnd, colStart:colEnd]
    return BigMatrix
    

##########################################################################
###################                  #####################################
################### community Matrix #####################################
###################                  #####################################
####### comNo communities of popSize individuals each ####################

def CommunitiesMatrix( popSize , comNo , upperBound):
    return MakeBistochastic( MakeBigMatrix(popSize, comNo, upperBound))

##########################################################################
###################                  #####################################
###################   Player Choice  #####################################
###################                  #####################################

def PickTwo(adj):
    n = np.shape(adj)[0]
    i = np.random.randint(0, n)
    j = np.random.choice(range(n), p = (adj[i,:]/np.norm(adj[i,:])).tolist())
    return [i,j]