"""
Functions to compute uniqueness tendency here.
This is written for one topic and just one pair of players.
We have to find dimensions of history to generalize it!
"""


import opiniongame.config as og_cfg
import opiniongame.IO as og_io
import opiniongame.coupling as og_coupling
import opiniongame.state as og_state
import opiniongame.opinions as og_opinions
import opiniongame.adjacency as og_adj
import opiniongame.selection as og_select
import opiniongame.potentials as og_pot
import opiniongame.core as og_core
import opiniongame.stopping as og_stop
import numpy as np

def findTendencies(config, state, players):

    """ This function takes history of evolution and adjacency matrix,
    and uses them to find uniqueness tendencies of each person.
    The function computes a certain variance for each individual's 
    random distribution of uniqueness tendency.and then samples from given distributions.
    """
    tendencies = None
    # find current opinion
    currentOpinions = np.copy(state.history[-1,:]).astype(float)

    # find neighbors of players
    speakerNeighbors = state.adj[players[0],:] > 0.
    hearerNeighbors =  state.adj[players[1],:] > 0.
    
    # pick up opinions of neighbors of players
    speakNeOpinions = np.multiply(speakerNeighbors, currentOpinions)
    hearNeOpinions = np.multiply(hearerNeighbors, currentOpinions)
    
    speakerDistaces = -np.abs((currentOpinions[players[0]] * speakerNeighbors) - speakNeOpinions)
    hearerDistances = -np.abs((currentOpinions[players[1]] * hearerNeighbors ) - hearNeOpinions)
	
	# Just pick up the d_ij's of neighbors. in above vectors there are some
	# extra zeros which came from nodes that are not neighbors of players.
	
	# and also, we cannot use np.nonzero, because there might be zeros in the vectors
	# that has come from neighbors with the same opinions!
	
    speakerDistaces = speakerDistaces[speakerNeighbors]
    hearerDistances = hearerDistances[hearerNeighbors]
    
    speakerVariance = state.uniqStrength * np.sum(np.power(np.e, speakerDistaces))
    hearerVariance  = state.uniqStrength * np.sum(np.power(np.e, hearerDistances))
    
    # speaker uniqueness
    tendencies[0] = np.random.normal(loc=0.0, scale=speakerVariance, size=None)
    # hearer uniqueness
    tendencies[1] = np.random.normal(loc=0.0, scale=hearerVariance, size=None)
    return tendencies