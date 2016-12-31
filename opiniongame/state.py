# object containing simulation state
import numpy as np

class worldStateClass:
    def __init__(self, adj, couplingWeights,
                 opinions):
        self.adj = adj
        self.couplingWeights = couplingWeights
        self.opinions = opinions

    def validate(self):
        # validation of data sizes
        print("WEIGHT SHAPE   : "+str(np.shape(self.couplingWeights)))
        print("OPINION SHAPE  : "+str(np.shape(self.opinions)))
        print("ADJACENCY SHAPE: "+str(np.shape(self.adj)))

        wPopsize = np.shape(self.couplingWeights)[0]
        wNtopics1 = np.shape(self.couplingWeights)[1]
        wNtopics2 = np.shape(self.couplingWeights)[2]

        oPopsize = np.shape(self.opinions)[0]
        oNtopics = np.shape(self.opinions)[1]

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
