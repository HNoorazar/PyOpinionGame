##########################################################################################
###################                                             ##########################
###################        Other Opinion Game models            ##########################
###################                                             ##########################



###################
###################  DeGroot
###################
###
### Row stochastic matrix to use in DeGroot model
### 
def converged_test_stochastic(matrix, threshold=0.0000001):
    trust_matrix = np.copy(matrix).astype(float);
    e1 = sum(abs(np.sum(trust_matrix , axis = 0) - 1));
    e2 = sum(abs(np.sum(trust_matrix , axis = 1) - 1));
    return (e2) > threshold
    
def one_step_stochastic(matrix):
    """ Here we will do one step towards
        Making a given matrix a bio-stochastic one        
        It does what OneStep does                         
    """
    # copy the input so that the original input is not changed.
    localMatrix = np.copy(matrix).astype(float);
    
    # Divide each row by sum of the entries in the given row.
    localMatrix = np.dot(np.diag(1/np.sum(localMatrix, axis=1)), localMatrix);
    return localMatrix

def make_row_stochastic(matrix):
    localMatrix = np.copy(matrix).astype(float);
    while (converged_test_stochastic(localMatrix)):
        localMatrix = one_step_stochastic(localMatrix);
    return localMatrix


def DeGroot_run(trust_matrix, initial_opinions, final_time=200):
    """
    input: trust_matrix: a row stochastic adjacency matrix whose entries
                         are the weights nodes puth on each other's opinion.

           initial_opinions: a vector of size (pop_size, 1)
           final_time: number of time steps
    outout: evolution of agent's opinions

    Note: There could be cases that convergence will not occur.
          for example if trust matrix is like the one given below,
          opinions will oscilate (So, at this point the only stopping criteria
          I chose is number of time steps!):

              [0 .5 .5]
          A = [1  0  0]
              [1  0  0]
    """
    pop_size = initial_opinions.shape[0]
    initial_opinions.resize(pop_size,)
    evolution = np.zeros((pop_size, final_time))
    evolution[:, 0] = initial_opinions
    for time in range(1, final_time):
        evolution[:, time] = np.dot(trust_matrix, evolution[:,time_step-1])
    return evolution






