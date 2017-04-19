import numpy as np
import opiniongame.stopping as og_stop

def pick_topic(nTopics):
    """
    Pick a topic from n possible topics.  Just a wrapper around NumPy randint.
    """
    return np.random.randint(nTopics)

def interaction_update(oldS, oldH, speaker_potential, hearer_potential, config, state, pPairs):
    """
    Input: opinion state of speaker and hearer, potential functions for each individual, learning rate
    Output: updated opinion state of each, and delta for each.
    """
    diff = oldS - oldH
    if config.uniqStrength == 0:
        speaker_delta = (config.learning_rate / 2.0) * speaker_potential(abs(diff)) * diff
        hearer_delta = (config.learning_rate / 2.0)  * hearer_potential(abs(diff))  * diff
    else:
        indvTendency = findTendencies(config, state, pPairs)
        speaker_delta = (config.learning_rate / 2.0) * speaker_potential(abs(diff)) * diff + indvTendency[0]
        hearer_delta = (config.learning_rate / 2.0)  * hearer_potential(abs(diff))  * diff + indvTendency[1]

    x = oldS - speaker_delta

    if x > 1.0:
        (newS, dS) = (1.0, 1.0 - oldS)
    elif x < 0.0:
        (newS, dS) = (0.0, 0.0 - oldS)
    else:
        (newS, dS) = (x, -speaker_delta)

    x = oldH + hearer_delta

    if x > 1.0:
        (newH, dH) = (1.0, 1.0 - oldH)
    elif x < 0.0:
        (newH, dH) = (0.0, 0.0 - oldH)
    else:
        (newH, dH) = (x, hearer_delta)

    return (newS, dS, newH, dH)

def handle_pair(config, state, ufunc, opinions, s, h):
    oldS = opinions[s, :]
    oldH = opinions[h, :]

    topic = pick_topic(config.ntopics)

    # TODO: fix this to support a family of potentials per individual
    speaker_func = ufunc.dPotential
    hearer_func = ufunc.dPotential

    (newS, dS, newH, dH) = interaction_update(oldS[topic], oldH[topic],
                                              speaker_func, hearer_func,
                                              config, state, [s, h])

    # Record all changes happening to each person, each topic, )
    wS = state.couplingWeights[s, topic, :]
    wH = state.couplingWeights[h, topic, :]
    chgS = 0.0
    chgH = 0.0
    
    for i in range(len(oldS)):
        newval = oldS[i] + dS * wS[i]
        if newval > 1:
            newval = 1
        elif newval < 0:
            newval = 0
        chgS = chgS + np.abs(opinions[s, i] - newval)
        opinions[s, i] = newval
        newval = oldH[i] + dH * wH[i]
        if newval > 1:
            newval = 1
        elif newval < 0:
            newval = 0
        opinions[h, i] = newval
        chgH = chgH + np.abs(opinions[h, i] - newval)

    return (opinions, chgS + chgH)

def one_step(config, state, ufunc):
    """ one_step entails updating as many pairs of individuals as allowed.
        input opinionO here is 2D
    """
    pairs = ufunc.selector(state.adj)

    curOpinions = state.currentOpinions()

    # Record all changes happening to each person, each topic, 
    # total_change is total change happening in one single time step!
    total_change = 0.0
    for i in range(pairs.shape[0]):
        (curOpinions, c) = handle_pair(config, state, ufunc, curOpinions, 
                                       pairs[i,0], pairs[i,1])
        total_change = total_change + abs(c)

    return (curOpinions, total_change, pairs)

#
# This function runs one single experiment.
#
def run_until_convergence(config, state, ufunc):
    # initialize history with 3D array at time =0
    state.initializeHistory()
    iterCount = 0
    terminate = False

    while not terminate:
        # take one step to produce the new set of opinions, change observed during the
        # step and pairs that interacted.
        (newOpinions, change, all_pairs) = one_step(config, state, ufunc)

        # compute change as the absolute change from the most recent step and
        # the previous.  IGNORE the change returned by one_step
        delt = np.sum(np.abs(state.history[-1]-state.history[-2]))

        state.appendToHistory(newOpinions)

        iterCount += 1
        userTermination = ufunc.stop(config, state, delt, iterCount)
        
        terminate = og_stop.polarizationStop(config, state, delt, iterCount) or userTermination

    state.history = np.delete(state.history, -1, axis=0)
    state.iterCount = iterCount
    return state
#
# This function finds uniqueness tendency of two players.
#
def findTendencies(config, state, players):

    """ This function takes history of evolution and adjacency matrix,
    and uses them to find uniqueness tendencies of each person.
    The function computes a certain variance for each individual's 
    random distribution of uniqueness tendency.and then samples from given distributions.
    """
    tendencies = np.zeros((2,1))
    # find current opinion
    currentOpinions = np.copy(state.history[-1,:]).astype(float)

    # find neighbors of players
    # returns a vector of size popSize where neighbors location is True, 
    # and non-neighbors are False
    speakerNeighbors = state.adj[players[0],:] > 0.
    speakerNeighbors = speakerNeighbors.reshape((len(speakerNeighbors),1))

    hearerNeighbors =  state.adj[players[1],:] > 0.
    hearerNeighbors = speakerNeighbors.reshape((len(hearerNeighbors),1))
    
    # pick up opinions of neighbors of players
    speakNeOpinions = np.multiply(speakerNeighbors, currentOpinions)
    speakerDistaces = -np.abs((currentOpinions[players[0]] * speakerNeighbors) - \
                                                              speakNeOpinions)

    hearNeOpinions = np.multiply(hearerNeighbors, currentOpinions)
    hearerDistances = -np.abs((currentOpinions[players[1]] * hearerNeighbors ) - \
                                                               hearNeOpinions)
	
	# Just pick up the d_ij's of neighbors. in above vectors there are some
	# extra zeros which came from nodes that are not neighbors of players.
	
	# and also, we cannot use np.nonzero, because there might be zeros in the vectors
	# that has come from neighbors with the same opinions!
	
    speakerDistaces = speakerDistaces[speakerNeighbors]
    speakerDistaces = speakerDistaces.reshape((len(speakerDistaces),1))

    hearerDistances = hearerDistances[hearerNeighbors]
    hearerDistances = speakerDistaces.reshape((len(hearerDistances),1))

    speakerVariance = (config.uniqstrength / (np.e - 1) ) * (-len(speakerDistaces) + \
                                      np.sum(np.power(np.e, 1 + speakerDistaces)))

    hearerVariance = (config.uniqstrength / (np.e - 1) ) * (-len(hearerDistances) + \
                                     np.sum(np.power(np.e, 1 + hearerDistances)))

    # speaker uniqueness force
    if speakerVariance == 0:
        tendencies[0] = 0
    else:
        tendencies[0] = np.random.normal(loc=0.0, scale=speakerVariance, size=None)
    # hearer uniqueness force
    if hearerVariance == 0:
        tendencies[1] = 0
    else:
        tendencies[1] = np.random.normal(loc=0.0, scale=hearerVariance, size=None)
    return tendencies
    
    
    