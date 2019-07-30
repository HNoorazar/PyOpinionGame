import numpy as np

def make_adj(pop_size, topology):
    """
    Create an adjacency matrix for a few specific types of graph given a population
    size and topology.  Supported types: star, full, grid.
    """

    adj = None

    if topology == 'star':
        adj = np.zeros((pop_size, pop_size))
        adj[:, [-1]] = 1
        adj[[-1], :] = 1
        adj[-1, -1] = 0
    elif topology == 'full':
        adj = np.zeros((pop_size, pop_size))
        adj = np.ones((pop_size, pop_size)) - np.identity(pop_size)
    elif topology == 'grid':
        adj = generateGridAdj(pop_size)
    return adj

#
# generate a grid graph.  computes the dimensions for the grid by
# calculating the prime factors (with duplication) of the population
# size and lets the nrows be the product of the odd elements of the
# factor list, and the ncols be the product of the even elements of
# the factor list.
#
def generateGridAdj(pop_size):
    def primes(n):
        primfac = []
        d = 2
        while d*d <= n:
            while (n % d) == 0:
                primfac.append(d)  # supposing you want multiple factors repeated
                n //= d
            d += 1
        if n > 1:
            primfac.append(n)
        return primfac

    factors = primes(pop_size)
    nrows = np.prod(factors[0::2])
    ncols = np.prod(factors[1::2])

    adj = np.zeros((nrows * ncols, nrows * ncols))

    for i in range(nrows):
        for j in range(ncols):
            idx = i*ncols + j
            if i-1 >= 0:
                nidx = (i-1)*ncols + j
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1
            if i+1 < nrows:
                nidx = (i+1)*ncols + j
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1
            if j-1 >= 0:
                nidx = i*ncols + (j-1)
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1
            if j+1 < ncols:
                nidx = i*ncols + (j+1)
                adj[idx, nidx] = 1
                adj[nidx, idx] = 1

    return adj

############################################################################
###################                          ###############################
################### Make Bistochastic Matrix ###############################
###################                          ###############################

### The following will test convergence!
def convergedTestBio(matrix):
    localMatrix = np.copy(matrix).astype(float);
    e1 = sum(abs(np.sum(localMatrix , axis = 0) - 1));
    e2 = sum(abs(np.sum(localMatrix , axis = 1) - 1));
    return (e1 + e2) > 0.000001
    
def OneStepBio(matrix):
    """ Here we will do one step towards
        Making a given matrix a bio-stochastic one        
        It does what OneStep does                         
        in Matt's code inside MakeBistochastic function.  
    """
    # copy the input so that the original input is not changed.
    localMatrix = np.copy(matrix).astype(float);
    
    # Divide each row by sum of the entries in the given row.
    localMatrix = np.dot(np.diag(1/np.sum(localMatrix, axis=1)), localMatrix);
    
    # Divide each column by sum of the elements in the given column.
    localMatrix = np.dot(localMatrix, np.diag(1/np.sum(localMatrix, axis=0)));
    localMatrix = np.triu(localMatrix) + np.transpose(np.triu(localMatrix,1));
    return localMatrix
    

def MakeBistochastic(matrix):
    localMatrix = np.copy(matrix).astype(float);
    while (convergedTestBio(localMatrix)):
        localMatrix = OneStepBio(localMatrix);
    return localMatrix

def MakeUpper(n):
    noElements = n * (n+1) / 2;
    size = 1 + (-1 + np.sqrt(1 + 8 * noElements)) / 2;
    upper = np.zeros([size, size]);
    upper[np.tril_indices(size, -1)] = np.arange(1, noElements+1)
    return upper        
    
def MakeBigMatrix(n, p, ubound):
    """ Make Big Matrix
        p communities of n individuals each
    """
    # inputs- p: number of communities, n: population size in a community
    #         ubound: upper boung for probability of interaction between communities.

    # This matrix will turn into a bio-stochastic later.
    
    BigMatrix = np.zeros(( n*p , n*p ));
    d = (1. / (n-1)) * np.ones((n,n)) - np.diag(np.ones(n) / (n-1));
    for rowCount in range(p):
        for colCount in range(rowCount, p):
            rowStart = rowCount * n;
            rowEnd   = rowCount * n + n;
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
                BigMatrix[rowStart:rowEnd, colStart:colEnd] = np.random.uniform(low = 0.0, 
                                                                                high = ubound, 
                                                                                size = (n, n));
                BigMatrix[colStart:colEnd, rowStart:rowEnd] = BigMatrix[rowStart:rowEnd, colStart:colEnd]
    return BigMatrix
    
def CommunitiesMatrix( popSize , comNo , upperBound):
    """ community Matrix
        comNo communities of popSize individuals each 
    """
    return MakeBistochastic( MakeBigMatrix(popSize, comNo, upperBound))



#################################################################################
#############                                          ##########################
############# Symmetric Doubly Stochastic adjacency    ##########################
############# with equal probabilities of interaction  ##########################


def MakeCommunityAdjDet(noCommunity, subPop, uppBound):
    " think about writing this in a clever way! leave it as brute force for now"
    " we are looking at just three communities"
    
    totoalPopulation = noCommunity * subPop
    deterministicAdj = np.zeros((totoalPopulation, totoalPopulation))
    diagonalBlock = np.ones((subPop, subPop)) * (1. / (subPop -1) ) - 
                    (1. / (subPop -1) ) *  np.identity(subPop)
    offDiagonalB = uppBound * np.ones((subPop, subPop))

    deterministicAdj[0:subPop, 0:subPop] = diagonalBlock
    deterministicAdj[subPop:2*subPop, 0:subPop] = offDiagonalB
    deterministicAdj[2*subPop: 3*subPop, 0:subPop] = offDiagonalB
    
    deterministicAdj[subPop:2*subPop, subPop: 2*subPop] = diagonalBlock
    deterministicAdj[0:subPop, subPop: 2*subPop] = offDiagonalB
    deterministicAdj[2*subPop:3*subPop, subPop: 2*subPop] = offDiagonalB
    
    deterministicAdj[2*subPop:3*subPop, 2*subPop:3*subPop] = diagonalBlock
    deterministicAdj[0:subPop , 2*subPop:3*subPop] = offDiagonalB
    deterministicAdj[subPop:2*subPop , 2*subPop:3*subPop] = offDiagonalB
    
    deterministicAdj = deterministicAdj / deterministicAdj[0,:].sum()
    return deterministicAdj

