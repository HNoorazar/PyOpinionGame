"""
Object containing dynamic simulation state.
"""

import numpy as np

class WorldState:
    def __init__(self, adj, couplingWeights, initialOpinions):
        self.adj = adj
        self.couplingWeights = couplingWeights
        self.initialOpinions = initialOpinions
        self.history = None

    def initializeHistory(self):
        hist = np.array([self.initialOpinions], copy=True)
        hist = np.concatenate((hist, [self.initialOpinions]), axis=0)
        self.history = hist

    def appendToHistory(self, newOpinions):
        self.history = np.concatenate((self.history, [newOpinions]), axis=0)

    def reset(self):
        self.history = None

    def currentOpinions(self):
        return self.history[-1]

    def validate(self):
        # validation of data sizes
        print("WEIGHT SHAPE   : "+str(np.shape(self.couplingWeights)))
        print("OPINION SHAPE  : "+str(np.shape(self.initialOpinions)))
        print("ADJACENCY SHAPE: "+str(np.shape(self.adj)))

        wPopsize = np.shape(self.couplingWeights)[0]
        wNtopics1 = np.shape(self.couplingWeights)[1]
        wNtopics2 = np.shape(self.couplingWeights)[2]

        oPopsize = np.shape(self.initialOpinions)[0]
        oNtopics = np.shape(self.initialOpinions)[1]

        aPopsize1 = np.shape(self.adj)[0]
        aPopsize2 = np.shape(self.adj)[1]

        if aPopsize1 != aPopsize2:
            raise ValueError("Adjacency matrix must be square.")
        if wNtopics1 != wNtopics2:
            raise ValueError("Per-topic weight matrix must be square.")
        if wPopsize != oPopsize or wPopsize != aPopsize1 or aPopsize1 != oPopsize:
            raise ValueError("Weight tensor, opinion state, and adjacency matrix disagree on population size.")
        if oNtopics != wNtopics1:
            raise ValueError("Weight tensor and opinion state disagree on topic count.")

        print("==> World state validation passed.")
        print("")
