# from numpy import array, matrix, arange, zeros, ones_like, zeros_like
# from numpy import mean, std, var, absolute, around
# from numpy import r_, c_, nonzero, unravel_index, sort, concatenate
# from numpy import int32, int64, float32, float64, isnan
# from numpy import exp, sin, cos, log, sqrt, diff, mod, diag
# from numpy import sum, floor, ceil, all
from numpy import *
import pandas as pd
from scipy.optimize import minimize_scalar
from itertools import product, permutations, repeat
from . import TrackingData as TD
from lap import lapmod, lapjv
import sys
if sys.version_info.major == 3:
    izip = zip
    imap = map
else:
    from itertools import izip, imap



def linkParticles(TrackingData, sigma=10., nprocs=1):
    #print('linking particles')
    P = _linkParticles(TrackingData, sigma, nprocs)
    #print('collecting trajectories')
    TrackingData.Data = _collectTrajectories(TrackingData, P)
    TrackingData.trajectoryStats()
    return TrackingData

def _linkParticles(TrackingData, sigma, nprocs):
    """find most likely links between pixels in adjacent frames.
       Uses linear assignment via the Hungarian algorithm.
    """
    LINKS = dict((int(t), {}) for t in arange(TrackingData.Nt))
    def getLocs(t):
        return TrackingData.particleSetGrouped[t][:, :3]
    def getProbs(t):
        return TrackingData.particleSetGrouped[t][:, 3]
    locs = izip(imap(lambda t: getProbs(t), arange(1, TrackingData.Nt)),
                imap(lambda t: getProbs(t), arange(TrackingData.Nt-1)),
                imap(lambda t: getLocs(t), arange(1, TrackingData.Nt)),
                imap(lambda t: getLocs(t), arange(TrackingData.Nt-1)))
    linker = imap(_linkFrame, izip(locs, repeat((sigma, TrackingData.zscale))))
    for t, MLlinks in enumerate(linker):
        if MLlinks.ndim == 2:
            Nlinks, _ = MLlinks.shape if MLlinks.size > 0 else (0, 0)
        else:
            Nlinks = 0
        LINKS[t+1]['Nlinks'] = Nlinks
        LINKS[t+1]['links'] = MLlinks
    return LINKS

def _linkFrame(args):
    P, params = args
    sigma = params[0]
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
            /(4.*sigma) - log(Pnow) - log(Pback)
        cutOff = 30.**2/(4.*sigma)
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
    # now = array(hungarian_solve(c))
    now, _ = array(lapjv(c, return_cost=False))
    back = arange(now.size)
    linkinds = (back < Nb)*(now < Nn)
    MLlinks = array([back[linkinds], now[linkinds]]).T
    return MLlinks

def _collectTrajectories(TrackingData, P):
    """Given links, collect all of the absolute positions for each particle.
       This step is just data extraction and formatting; no statistical
       computations happen here."""
    dataOut = []
    pn = 0
    #keep track of what particle positions we've already extracted
    collectedParticles = dict((k, []) for k in arange(TrackingData.Nt))
    def pSet(t, n):
        return TrackingData.particleSetGrouped[t][n, :3]
    def getRad(t, n):
        return TrackingData.particleSetGrouped[t][n, 4]
    def getProb(t, n):
        return TrackingData.particleSetGrouped[t][n, 3]
    def getI(t, n):
        return (TrackingData.particleSetGrouped[t][n, 5],
            TrackingData.particleSetGrouped[t][n, 6],
            TrackingData.particleSetGrouped[t][n, 7])
    def loopFn(k, pn, b_in):
        def loopflag(k, b_in):
            if k<TrackingData.Nt-1:
                oneFrame = (P[k]['links'].size>0) \
                    and (b_in in P[k]['links'][:, 0])
                return oneFrame
            if k==TrackingData.Nt-1:
                return (P[k]['links'].size>0) and (b_in in P[k]['links'][:, 0])
            else:
                return False
        while loopflag(k, b_in):
            if (P[k]['links'].size>0 and (b_in in P[k]['links'][:, 0])):
                inds = P[k]['links'][:, 0] == b_in
                ind = arange(P[k]['Nlinks'])[inds][0]
                b, n = P[k]['links'][ind]
                if n in collectedParticles[k]:
                    return
                xnow, ynow, znow = pSet(k, n)
                radnow = getRad(k, n)
                pnow = getProb(k, n)
                IbgNow, IpeakNow, SNRnow = getI(k, n)
                dataOut.append({'particle': pn, 'frame': k,
                                'x': xnow, 'y': ynow, 'z': znow,
                                'p': pnow, 'r': radnow,
                                'Ibg':IbgNow, 'Ipeak':IpeakNow,
                                'SNR':SNRnow})
                collectedParticles[k].append(n)
                k += 1
                b_in = n
    ###############################
    for k in arange(1, TrackingData.Nt):
        for j in arange(P[k]['Nlinks']):
            b, n = P[k]['links'][j]
            if (b in collectedParticles[k-1]) or (n in collectedParticles[k]):
                continue
            xback, yback, zback = pSet(k-1, b)
            xnow, ynow, znow = pSet(k, n)
            radback, radnow = getRad(k-1, b), getRad(k, n)
            pback, pnow = getProb(k-1, b), getProb(k, n)
            IbgBack, IpeakBack, SNRback = getI(k-1, b)
            IbgNow, IpeakNow, SNRnow = getI(k, n)
            dataOut.append({'particle': pn,
                            'frame': k-1,
                            'x': xback,
                            'y': yback,
                            'z': zback,
                            'p': pback,
                            'r': radback,
                            'Ibg': IbgBack,
                            'Ipeak': IpeakBack,
                            'SNR': SNRback})
            dataOut.append({'particle': pn,
                            'frame': k,
                            'x': xnow,
                            'y': ynow,
                            'z': znow,
                            'p': pnow,
                            'r': radnow,
                            'Ibg': IbgNow,
                            'Ipeak': IpeakNow,
                            'SNR': SNRnow})
            collectedParticles[k-1].append(b)
            collectedParticles[k].append(n)
            loopFn(k+1, pn, n)
            pn += 1
    names = ['particle', 'frame', 'x', 'y', 'z', 'p',
        'r', 'Ibg', 'Ipeak', 'SNR']
    dataOut = pd.DataFrame(dataOut, columns=names) \
        .sort_values(['particle', 'frame'])
    return dataOut
