"""
Potential function derivative definitions.

In this code potentials are defined as closures.
This allows one to instantate a parameterized potential once and pass around the
resulting function as a closure without needing to thread the potential parameters
through the code.
"""
from math import exp, pi

def createTent(center, leftSlope, rightSlope):
    def tent(x):
        if x >= 0 and x < center:
            return leftSlope
        elif x >= center and x <= 1.0:
            return rightSlope
        else:
            return 0.0
    return tent

## examples

#indTent = createTent(0.3, 10./3, 0.)
#midTent = createTent(0.5, 2., -2.)
#lsTent  = createTent(0.3, 10./3, -10./7)
#rsTent  = createTent(0.8, 10./8, -5.)

def flatPotential(x):
    return 0.0

def flatTopTent(flatRegion, leftSlope, rightSlope):
    def flatTent(x):
        if x >= 0 and x < flatRegion[0]:
            return leftSlope
        elif x >= flatRegion[0] and x < flatRegion[1]:
            return 0.0
        elif x >= flatRegion[1] and x <= 1:
            return rightSlope
        else:
            return 0.0

    return flatTent

def gaussian(x):
    mean = .5
    z = (2 * pi)**(.5)
    if x >= 0 and x <= 1:
        return ((mean - x)/z) * exp(-(x - mean)**2/2.0)
    else:
        return 0.0
        