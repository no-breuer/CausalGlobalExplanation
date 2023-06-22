from module_data_gen import *
from scipy.stats import bernoulli
import numpy as np
import scipy as sp

'''
This code implements the SCM. 
'''

# Fix the random seed
#np.random.seed(1234)

# def gen_U(n):
#     p = 0.4
#     var_U = bernoulli.rvs(p, size=n)
#     return(var_U)

def gen_V1(n):
    #p = 0.8
    #var_UV1 = bernoulli.rvs(p, size=n)
    var_UV1 = np.random.normal(1.5, 1, size=n)
    return (var_UV1)

def gen_V3(n):
    #p = 0.4
    #var_UV3 = bernoulli.rvs(p, size=n)
    # var_V3 = np.logical_or(var_UV3, U) * 1
    var_UV3 = np.random.normal(0.5, 2, size=n)
    return (var_UV3)

def gen_V2(n,V1,V3):
    #p = 0.3
    #var_UV2 = bernoulli.rvs(p, size=n)
    var_UV2 = np.random.uniform(0, 0.1, size=n)
    #var_V2 = np.logical_or( np.logical_and(V1, V3), var_UV2 ) * 1
    var_V2 = V1 + V3 + var_UV2
    return (var_V2)

def gen_Y(n,V1,V2,V3):
    # conv_V1 = 2 * V1 - 1
    # conv_V2 = 2 * V2 - 1
    # conv_V3 = 2 * V3 - 1
    UY = np.random.uniform(0, 0.1, size=n)
    var_Y  = 3*V1 + 0.4*V2 + V3 + UY
    return(var_Y)

def SCM(n,v1=-100,v2=-100,v3=-100,seednum=1):
    np.random.seed(seednum)
    if v1 == -100:
        V1 = gen_V1(n)
    else:
        V1 = np.repeat(v1,n)

    if v3 == -100:
        V3 = gen_V3(n)
    else:
        V3 = np.repeat(v3,n)

    if v2 == -100:
        V2 = gen_V2(n, V1, V3)
    else:
        V2 = np.repeat(v2,n)
    V = np.matrix((V1, V2, V3)).T
    Y = gen_Y(n, V1, V2, V3)
    return([V,Y])


if __name__ == '__main__':
    n = 1000
    V1 = gen_V1(n)
    V3 = gen_V3(n)
    V2 = gen_V2(n)
    Y = gen_Y(n, V1, V2, V3)



