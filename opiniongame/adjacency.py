import numpy as np

#
# This function would create adjacency matrix for very limited types of graphs.
#
def make_adj(pop_size, topology):
    Adj = np.zeros(( pop_size, pop_size ))
    if topology == 'star':
        Adj[:,[-1]] = 1
        Adj[[-1],:] = 1
        Adj[-1,-1] = 0
    elif topology == 'full':
        Adj = np.ones((pop_size, pop_size)) - np.identity(pop_size)
    return Adj

#
# generate a grid graph.  computes the dimensions for the grid by
# calculating the prime factors (with duplication) of the population
# size and lets the nrows be the product of the odd elements of the
# factor list, and the ncols be the product of the even elements of
# the factor list.
#
def generateGridAdj(popsize):
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

    factors = primes(popsize)
    nrows = np.prod(factors[0::2])
    ncols = np.prod(factors[1::2])

    Adj = np.zeros((nrows * ncols,nrows * ncols))
    
    
    for i in range(ncols):
        for j in range(nrows):
            k = np.ravel_multi_index((i,j), dims=(ncols, nrows), order='F')
            if i > 0:
                ii = i-1
                jj = j
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
                        
            if i<ncols-1:
                ii=i+1
                jj=j
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
            
            if j>0:
                ii=i
                jj=j-1
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
            if j<nrows-1:
                ii=i
                jj=j+1
                kk = np.ravel_multi_index((ii,jj), dims=(ncols, nrows), order='F')
                Adj[k,kk] = 1
        
    return Adj