import numpy as np
import opiniongame.stopping as og_stop

def pick_topic(nTopics):
    """
    Pick a topic from n possible topics.  Just a wrapper around NumPy randint.
    """
    return np.random.randint(nTopics)

def interaction_update(oldS, oldH, speaker_potential, hearer_potential, learning_rate):
    """
    Input: opinion state of speaker and hearer, potential functions for each individual, learning rate
    Output: updated opinion state of each, and delta for each.
    """
    diff = oldS - oldH

    speaker_delta = (learning_rate / 2.0) * speaker_potential(abs(diff)) * diff
    hearer_delta = (learning_rate / 2.0) * hearer_potential(abs(diff)) * diff

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

def handle_pair(statConfigs, dynState, ufunc, currentOpinionMatrix, s, h):
    oldS = currentOpinionMatrix[s, :]
    oldH = currentOpinionMatrix[h, :]

    topic = pick_topic(statConfigs.ntopics)

    # TODO: fix this to support a family of potentials per individual
    speaker_func = ufunc.dPotential
    hearer_func = ufunc.dPotential

    (newS, dS, newH, dH) = interaction_update(oldS[topic], oldH[topic],
                                              speaker_func, hearer_func,
                                              statConfigs.learning_rate)

    # Record all changes happening to each person, each topic, )
    wS = dynState.couplingWeights[s, topic, :]
    wH = dynState.couplingWeights[h, topic, :]
    chgS = 0.0
    chgH = 0.0
    
########    Vectorized of the next part      ##########################   
########    Why vectorization does not work?
#    newval = oldS + dS * wS
#    newval[newval > 1 ] = 1
#    newval[newval < 0 ] = 0
#    chgS = np.sum( np.abs(opinions[s,:] - newval) )

#    newval = oldH + dH * wH
#    newval[newval > 1 ] = 1
#    newval[newval < 0 ] = 0
#    chgH = np.sum( np.abs(opinions[h,:] - newval) )
##################################################################
    for i in range(len(oldS)):
        newval = oldS[i] + dS * wS[i]
        if newval > 1:
            newval = 1
        elif newval < 0:
            newval = 0
        chgS = chgS + np.abs(currentOpinionMatrix[s,i] - newval)
        currentOpinionMatrix[s,i] = newval
        newval = oldH[i] + dH * wH[i]
        if newval > 1:
            newval = 1
        elif newval < 0:
            newval = 0
        currentOpinionMatrix[h,i] = newval
        chgH = chgH + np.abs( currentOpinionMatrix[h,i] - newval )       
##################################################################        

    return (currentOpinionMatrix, chgS + chgH)    

def one_step(config, game_state, ufunc, opinionsO):
    """ one_step entails updating as many pairs of individuals as allowed.
        input opinionO here is 2D
    """
    pairs = ufunc.selector(game_state.adj)

    # Record all changes happening to each person, each topic, 
    # total_change is total change happening in one single time step!
    total_change = 0.0
    for i in range(pairs.shape[0]):
        (opinionsO, c) = handle_pair(config, game_state, ufunc, opinionsO, pairs[i,0], pairs[i,1])
        total_change = total_change + c

    # Output opinions has to be 3D to be used by "concatenate"
    opinions3d = np.zeros((1, np.shape(opinionsO)[0], np.shape(opinionsO)[1] ))
    opinions3d[0,:,:] = opinionsO
    return (opinions3d, total_change, pairs)



def run_until_convergence(config, state, ufunc):
    print("==> RUN_UNTIL_CONVERGENCE")

    # initialize history with 3D array at time =0
    hist = np.array([state.opinions], copy=True)
    hist = np.concatenate((hist, [state.opinions]), axis=0)
    state.history = hist

    iterCount = 0

    terminate = False

    while not terminate:
        # take one step to produce the new set of opinions, change observed during the
        # step and pairs that interacted.
        (newOpinions, change, all_pairs) = one_step(config, state, ufunc, state.history[-1])

        print(str(change))

        state.history = np.concatenate((state.history, newOpinions), axis=0)
        iterCount += 1

        terminate = ufunc.stop(config, state, change, iterCount)

    state.history = np.delete(state.history, -1, axis=0)

    return state
