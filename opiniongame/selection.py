import numpy as np

# code related to selection of individuals during a game
# And also this just works on full graph. look at number of games!
# (may be that's the only thing need to change)

# pairs are numpy arrays of size (noGames * 2)

def FastPairSelection(Adj):
    pop_size = int(np.shape(Adj)[1])
    ngames = int(pop_size/2)
    pairs = np.zeros(shape=(ngames, 2), dtype=int)

    # copy the adjacency matrix so we can destructively update it
    # during the algorithm.
    M = np.array(Adj, copy=True)

    # list of available individuals.
    avail = np.arange(pop_size)

    # translation array mapping individual IDs to their index in the
    # avail list.  necessary as individuals in avail get shuffled around.
    indices = np.arange(pop_size)

    # number of available individuals.  this shrinks.
    numavail = pop_size
    ii=0
    while numavail > 1:
        # pick a number from 0 to numavail-1
        i = np.random.choice(numavail)

        # speaker is the available individual at index i
        speaker = avail[i]

        # copy last individual in the list to where speaker was
        avail[i] = avail[numavail-1]

        # redirect copied individual index to point at new location
        indices[avail[i]] = i

        # speaker is gone, so its index points to -1
        indices[speaker] = -1

        # decrement available count
        numavail = numavail - 1

        # zero out speaker row in adjacency matrix 
        M[speaker, :] = 0

        # extract list of potential hearers
        potentialHearers = np.where(M[:, speaker] == 1)

        if np.size(potentialHearers) > 0:
            # pick one of the hearers
            hearerWhich = np.random.choice(len(potentialHearers[0]))
            hearer = potentialHearers[0][hearerWhich]

            # remember the speaker and hearer
            pairs[ii, 0] = speaker
            pairs[ii, 1] = hearer

            # zero out hearer row in adjacency matrix
            M[hearer, :] = 0

            # do same trick as above for speaker, and shuffle the hearer
            # out of the array of available indices.
            hearerIndex = indices[hearer]
            avail[hearerIndex] = avail[numavail-1]
            indices[avail[hearerIndex]] = hearerIndex
            indices[hearer] = -1
            ii = ii+1
            numavail = numavail-1
    pairs = pairs[~np.all(pairs == 0, axis=1)]
    return pairs

""" Function for disributing probabilities between available agents, to get
    conditional probabilities.
"""

def disrtibuteProbabilities(vecProp):
    """ Distribute the missing probability between 
        available agents
    """
    pSum = np.sum(vecProp)
    return ((1 - pSum)/pSum) * vecProp + vecProp
    
    

#
# Pick a pair from a weighted adjacency matrix.
#
def PickTwoWeighted(weightedAdj):
    # copy the weightedAdj matrix so we can destructively update it
    # during the algorithm.
    pairs = np.zeros(shape=(1,2), dtype=int)
    adj = np.copy(weightedAdj).astype(float)
    n = np.shape(adj)[0]

    # choose first player
    pairs[0,0] = np.random.randint(0, n)
    
    # find number of neighbors of speaker
    noNeighbors = np.count_nonzero(adj[pairs[0,0],:])

    rowSum = np.sum(adj[pairs[0,0],:])
    if rowSum == 1.0:
        normalizedVector = adj[pairs[0,0],:]
    #
    # this would not work if minimum entry in the given row, is smaller than
    # diff. because then that entry would become negative!
    #
    elif rowSum > 1.0:  
        diff = rowSum - 1.
        normalizedVector = adj[pairs[0,0],:] - ( diff/noNeighbors )
        normalizedVector[pairs[0,0]] = 0
    else:
        diff = 1. - rowSum
        normalizedVector = adj[pairs[0,0],:] + ( diff/noNeighbors )
        normalizedVector[pairs[0,0]] = 0
    pairs[0,1] = np.random.choice(range(n), p = normalizedVector.tolist())
    return pairs

"""
 This function would pick max number of players from a weighted matrix of a full graph.
 will be developed later for other adjacencies!!!
"""
##############
############## This function is written for future work.
############## It may need a little bit of attention to details.
############## 
def pickMaxPlayersWeightedAdj(weightedAdjacency):
    pop_size = int(np.shape(weightedAdjacency)[1])
    ngames = int(pop_size/2)
    pairs = np.zeros(shape=(ngames, 2), dtype=int)
    
    # copy the adjacency matrix so we can destructively update it
    # during the algorithm.
    M = np.array(weightedAdjacency, copy=True)
    
    # list of available individuals.
    avail = np.arange(pop_size)

    # translation array mapping individual IDs to their index in the
    # avail list.  necessary as individuals in avail get shuffled around.
    indices = np.arange(pop_size)

    # number of available individuals.  this shrinks.
    numavail = pop_size
    ii=0
    while numavail > 1:
        # pick a number from 0 to numavail-1
        i = np.random.choice(numavail)

        # speaker is the available individual at index i
        speaker = avail[i]

        # copy last individual in the list to where speaker was
        avail[i] = avail[numavail-1]

        # redirect copied individual index to point at new location
        indices[avail[i]] = i

        # speaker is gone, so its index points to -1
        indices[speaker] = -1

        # decrement available count
        numavail = numavail - 1

        # zero out speaker row in adjacency matrix
        M[speaker, :] = 0

        # extract list of potential hearers
        potentialHearers = np.where(M[:, speaker] == 1)

        if np.size(potentialHearers) > 0:
            # find number of neighbors of speaker
            noNeighbors = np.count_nonzero(M[:, speaker])
            # find sum of probabilities 
            colSum = np.sum(M[:, speaker])

            # pick one of the hearers
            hearerWhich = np.random.choice(len(potentialHearers[0]))
            hearer = potentialHearers[0][hearerWhich]

            # remember the speaker and hearer
            pairs[ii, 0] = speaker
            pairs[ii, 1] = hearer

            # zero out hearer row in adjacency matrix
            M[hearer, :] = 0

            # do same trick as above for speaker, and shuffle the hearer
            # out of the array of available indices.
            hearerIndex = indices[hearer]
            avail[hearerIndex] = avail[numavail-1]
            indices[avail[hearerIndex]] = hearerIndex
            indices[hearer] = -1
            ii = ii+1
            numavail = numavail-1
    pairs = pairs[~np.all(pairs == 0, axis=1)]
    return pairs



