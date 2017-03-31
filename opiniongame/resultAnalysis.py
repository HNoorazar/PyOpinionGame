import numpy as np

def determineConsensus(oneHistory):
        if np.max(oneHistory[-1]) - np.min(oneHistory[-1]) < .001:
        return True
