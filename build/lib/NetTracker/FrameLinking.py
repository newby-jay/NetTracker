from __future__ import division
from __future__ import print_function
from numpy import *
import pandas as pd
from itertools import product, permutations, repeat
# from multiprocessing import Pool
# from contextlib import closing
from Hungarian import hungarian_solve
import sys
if sys.version_info.major == 3:
    izip = zip
    imap = map
else:
    from itertools import izip, imap

def linkFrame(args):
    P, params = args
    D = params[0]
    zscale = params[1]
    pnow, pback, rnow, rback = P
    if rnow.size == 0 or rback.size == 0:
        return array([[]])
    Nb, _ = rback.shape
    Nn, _ = rnow.shape
    def make_c():
        Xnow, Xback = meshgrid(rnow[:, 0], rback[:, 0])
        Ynow, Yback = meshgrid(rnow[:, 1], rback[:, 1])
        Znow, Zback = meshgrid(rnow[:, 2], rback[:, 2])
        Pnow, Pback = meshgrid(pnow, pback)
        logL =  (
            (Xnow - Xback)**2
            + (Ynow - Yback)**2
            + zscale**2*(Znow - Zback)**2) \
            /(4.*D) - log(Pnow) - log(Pback)
        cutOff = 30.**2/(4.*D)
        logL[logL > cutOff] = inf
        Nb, Nn = logL.shape
        death = diag(-log(1.-pback))
        birth = diag(-log(1.-pnow))
        death[death==0] = inf
        birth[birth==0] = inf
        #min1 = min(pback.min(), pnow.min()) # x < -log(1-x), x\in(0, 1)
        #R = min(min1, logL.min())*ones((Nn, Nb))
        R = zeros((Nn, Nb))
        R[logL.T > cutOff] = inf
        c = r_[c_[logL, death],
               c_[birth,    R]]
        return float32(c)
    c = make_c()
    assert sum(isnan(c)) == 0
    now = array(hungarian_solve(c))
    back = arange(now.size)
    linkinds = (back<Nb)*(now<Nn)
    MLlinks = array([back[linkinds], now[linkinds]]).T
    return MLlinks
