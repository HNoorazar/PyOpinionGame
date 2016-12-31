import scipy.io as sio

def loadNamedMatrix(filename, name):
    try:
        mat = sio.loadmat(filename)
        m = mat[name]
    except:
        print("ERROR: could not load matrix "+filename)
        m = None
    return m

