"""
Functions to compute uniqueness tendency here.
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



def tendencies = findVariances(config, state):
    """ This function takes history of evolution and adjacency matrix,
    and uses them to find uniqueness tendencies of each person.
    The function computes a certain variance for each individual's 
    random distribution uniqueness tendency.
    """     

    
