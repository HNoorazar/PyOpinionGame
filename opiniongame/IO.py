import scipy.io as sio
import numpy as np
import h5py

def loadNamedMatrix(filename, name):
    """
    Read a MATLAB-formatted matrix from a file, and extract the
    given named variable and return its value.
    """
    try:
        mat = sio.loadmat(filename)
        m = mat[name]
    except:
        print("ERROR: could not load matrix "+filename)
        m = None
    return m

def saveMatrix(filename, matDict):
    """
    Write a MATLAB-formatted matrix file given a dictionary of
    variables.
    """
    try:
        sio.savemat(filename, matDict)
    except:
        print("ERROR: could not write matrix file "+filename)


def saveMatrixH5(filename, history, config, state):
    try:
        hf = h5py.File('data.hdf5', 'w')
        hf.create_dataset(filename, data=fDict)

        """
         We could also do:
            with h5py.File("mytestfile.hdf5", "w") as f:
            dset = f.create_dataset("mydataset", (100,), dtype='i')
        """
        hf.close()
    except:
        print("ERROR: could not write matrix file "+filename)

