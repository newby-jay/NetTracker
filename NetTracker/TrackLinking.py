from __future__ import division
from __future__ import print_function
from numpy import *
import pandas as pd
from scipy.optimize import minimize_scalar
from itertools import product, permutations, repeat
# from multiprocessing import Pool
# from contextlib import closing
import TrackingData as TD
from Hungarian import hungarian_solve
from lap import lapmod
import sys
if sys.version_info.major == 3:
    izip = zip
    imap = map
else:
    from itertools import izip, imap

class SparseArrayFloat32:
    def __init__(self):
        self.I = zeros((0), 'int64')
        self.J = zeros((0), 'int64')
        self.A = zeros((0), 'float32')
        self.size = 0
    def _validateData(self, I, J, A):
        assert all(I.shape == J.shape)
        assert all(J.shape == A.shape)
        assert I.ndim == 1
        assert all(I >= 0)
        assert all(J >= 0)
        assert len(set(zip(I, J))) == I.size, 'indices must be unique'
        assert all(isfinite(I))
        assert all(isfinite(J))
        assert all(isfinite(A))
        return True
    def addData(self, I, J, A):
        I = array(I, 'int64')
        J = array(J, 'int64')
        A = float32(A)
        assert all(I.shape == J.shape)
        assert all(J.shape == A.shape)
        if I.size == 0:
            return self
        # self._validateData(I, J, A)
        if self.size == 0:
            self.I, self.J, self.A = I, J, A
        else:
            self.I = r_[self.I, I]
            self.J = r_[self.J, J]
            self.A = r_[self.A, A]
        self.size = self.I.size
        self._validateData(self.I, self.J, self.A)
        return self
    def convert(self):
        inds = lexsort((self.I, self.J))
        kk = int32(self.I[inds])
        jj = int32(self.J[inds])
        vals = float32(self.A[inds])
        first = int32(r_[0, arange(self.size)[r_[0, diff(jj)] != 0], vals.size])
        assert kk.max() == first.size - 2
        assert kk.min() == 0
        assert jj.max() == first.size - 2
        assert jj.min() == 0
        assert all(first >= 0)
        assert all(first <= vals.size)
        assert len(set(first)) == first.size
        return float64(vals), kk, first

def prepareTrackData(trackData):
    tracks = trackData.groupby('particle')
    Ntracks = tracks.ngroups
    assert Ntracks > 1
    ## starts: t, x, y, z
    ## ends: t, x, y, z
    ## lengths (number of observations)
    ##
    starts = zeros((Ntracks, 4), float64)
    ends = zeros((Ntracks, 4), float64)
    lengths = zeros((Ntracks), int64)
    particleNumbers = zeros((Ntracks), int64)
    mobilities = zeros((Ntracks), float64)
    for n, (p, g) in enumerate(tracks):
        track = array(g[['frame', 'x', 'y', 'z']])
        starts[n] = track[0]
        ends[n] = track[-1]
        lengths[n] = track.shape[0]
        particleNumbers[n] = p
        dt = diff(track[:, 0])
        dxyz = diff(track[:, 1:], axis=0)
        dr_squared = (dxyz**2).sum(axis=1)
        assert dt.shape == dr_squared.shape
        mobilities[n] = (dr_squared/dt).mean()
    assert all(array(particleNumbers) == arange(Ntracks))
    return starts, ends, lengths, mobilities

def makeCostMatrix(Data, tLinkScale, distMax, birth, death):
    """Generate cost matrix for track linking.
    Data: output from `prepareTrackData()`.
    tLinkScale: max number of frames between two tracks (end to start).
    distMax: max distance (between end and start of two tracks) to consider
    linking.
    birth: parameter, weight for cost of not linking the start of a track.
    death: parameter, weight for cost of not linking the end of a track."""
    ## [[(Ntrack, Ntracks): 0.5*L, (Ntrack, Ntracks): deaths],
    ## [(Ntrack, Ntracks): births, (Ntrack, Ntracks): 0.5*L.T]]
    starts, ends, lengths, mobilities = Data
    Ntracks = lengths.size
    C = SparseArrayFloat32()
    Nt = ends[:, 0].max()
    maxTimeSkip = int(0.15*Nt)
    I, J, L = [], [], []
    for e in arange(Ntracks):
        te = ends[e, 0]
        xe = ends[e, 1:]
        D = max(1e-2, 2.*mobilities[e])
        for s in arange(Ntracks):
            if s == e:
                continue
            ts = starts[s, 0]
            xs = starts[s, 1:]
            if not te < ts <= te + maxTimeSkip:
                continue
            d = ((xe - xs)**2).sum()
            dt = ts - te
            assert isfinite(d) and d >= 0
            if sqrt(d) > distMax:
                continue
            assert D > 0 and isfinite(D)
            assert dt > 0 and isfinite(dt)
            l = d/D/dt + dt/tLinkScale
            I.append(e)
            J.append(s)
            L.append(l)
    I, J, L = array(I), array(J), array(L)
    C.addData(I, J, L)
    C.addData(Ntracks + J, Ntracks + I, 0*L)
    ## deaths
    I = arange(Ntracks)
    J = Ntracks + I
    lengthCost = 1. + 0.25*log(lengths)
    C.addData(I, J, death*lengthCost)
    C.addData(J, I, birth*lengthCost)
    return C

def collectGroups(E, S):
    Nlinks = S.size
    r = arange(Nlinks)
    Groups = []
    collectedParticles = []
    for j in arange(Nlinks):
        e, s = E[j], S[j]
        if e in collectedParticles or s in collectedParticles:
            continue
        group = [e, s]
        collectedParticles.append(e)
        collectedParticles.append(s)
        while s in E:
            ind = r[s == E][0]
            s = S[ind]
            if s in collectedParticles:
                continue
            group.append(s)
            collectedParticles.append(s)
        Groups.append(group)
    return Groups

def renumberTracks(particles, Ntracks, trackGroups):
    pMap = arange(Ntracks)
    for group in trackGroups:
        pTemp = array(group).min()
        for p in group:
            pMap[p] = pTemp
    pOld = list(set(pMap))
    particlesMap = dict((pOld[n], n) for n in arange(len(pOld)))
    pNew = array([particlesMap[pMap[p]] for p in particles])
    return pNew

def linkTracks(trackData, tLinkScale=30, distMax=50, birth=2., death=2.):
    trackData.trajectoryStats(output=False)
    Ntracks = trackData.Nparticles
    if Ntracks < 2 or trackData.Nt < 20:
        return trackData
    Data = prepareTrackData(trackData.Data)
    Cs = makeCostMatrix(Data, tLinkScale, distMax, birth, death)
    # C = empty((2*Ntracks, 2*Ntracks), float32)
    # C[:] = inf
    # C[Cs.I, Cs.J] = Cs.A
    # start = array(hungarian_solve(C))
    vals, kk, offsets = Cs.convert()
    _, start = array(lapmod(2*Ntracks, vals, offsets, kk, return_cost=False))
    end = arange(start.size)
    linkinds = (start < Ntracks) & (end < Ntracks)
    S, E = start[linkinds], end[linkinds]
    if S.size == 0:
        return trackData
    trackGroups = collectGroups(E, S)
    pNew = renumberTracks(
        array(trackData.Data.particle),
        Ntracks,
        trackGroups)
    newTracks = trackData.Data.copy().assign(particle=pNew)
    out = TD.TrackingData(shape=trackData.shape).setData(newTracks)
    # out.setDetections(trackData.detectionData.copy())
    return out
