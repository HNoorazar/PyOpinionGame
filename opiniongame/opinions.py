import numpy as np

#   
#   initialize opinions
#   1 is added there to use contacenate thing later!!!   
def initialize_opinions( pop_size, ntopics ):
    return np.random.rand(pop_size, ntopics)