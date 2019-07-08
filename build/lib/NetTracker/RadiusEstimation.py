from __future__ import division
from __future__ import print_function
from numpy import *
import pandas as pd
from scipy.optimize import minimize_scalar
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

def estimateRadii(args):
    locs, volume = args
    volume = float64(volume)
    if locs.size == 0:
        return array([]), array([]), array([]), array([])
    Nlocs, _ = locs.shape
    radii = zeros(Nlocs)
    Ibg = zeros(Nlocs)
    Ipsf = zeros(Nlocs)
    SNR = zeros(Nlocs)
    rmax = 20.
    s = arange(-rmax, rmax+1)
    near = array([xn for xn in product(s, s)
        if sqrt((array(xn)**2).sum()) <= rmax], dtype='int')
    N1, N2 = volume[..., 0].shape
    def estimate(xave, frame):
        xinds = around(xave).astype(int)
        patchInds = array([ij for ij in xinds + near
                           if ij[1] >= 0 and ij[1] <= N1-1
                           and ij[0] >= 0 and ij[0] <= N2-1
                           and frame[ij[1], ij[0]] >= 0])
        if patchInds.ndim != 2:
            return 0., 0., 0., 0.
        I = frame[patchInds[:, 1], patchInds[:, 0]]
        EI = mean(I)
        d2 = sum((xave - patchInds)**2, axis=1)
        def minFun(de):
            C = de**2/2.
            zeta = exp(-0.5*(d2/C)**2)
            Ezeta = mean(zeta)
            B = max(10.0, (mean(I*zeta) - EI*Ezeta)/var(zeta))
            A = EI - B*Ezeta
            return mean((A + B*zeta - I)**2)
        res = minimize_scalar(
            minFun,
            bounds=(2, 10),
            method='bounded',
            tol=1e-5)
        C = res.x**2/2.
        zeta = exp(-0.5*(d2/C)**2)
        Ezeta = mean(zeta)
        # B = max(10.0, (mean(I*zeta) - EI*Ezeta)/var(zeta))
        B = (mean(I*zeta) - EI*Ezeta)/var(zeta)
        A = EI - B*Ezeta
        # (A + B*zeta - I)^2
        localSTD = std(A + B*zeta - I)
        if localSTD > 0:
            snr = absolute(B)/localSTD
        elif B == 0:
            snr = 0
        else:
            snr = inf
        return res.x, A, A + B, snr
    ####################
    for n in arange(Nlocs):
        xave = locs[n]
        zn = int(around(xave[2]))
        frame = volume[..., zn]
        radii[n], Ibg[n], Ipsf[n], SNR[n] = estimate(xave[:2], frame)
    return radii, Ibg, Ipsf, SNR

class EstimateRadii:

    def __init__(self, localizations):
        self.particleSet = localizations
        if len(filename)>0:
            try:
                self.Data = pd.read_csv(filename+' (tracks).csv', index_col=0)
                self.Nt = self.Data.frame.max()
                self.trajectoryStats(output=True)
            except:
                print('particle paths not found')
            try:
                self.setDetections(
                    pd.read_csv(
                        filename+' (localizations).csv',
                        index_col=0)
                    )
            except:
                print('particle set not found')
                raise
            return None
        elif len(shape)>0:
            self.Nt, self.Ny, self.Nx, self.Nz = shape
            self.shape = shape
        else:
            return None
    def _estimateAllRadii(self, vid, nprocs, ds):
        """Gaussian fit to local region, limit 15 pixel radius.
           Minimizes (A + B*exp(-0.5*(d_ij/r)**2) - I_ij)**2.
           Output is r.
        """
        for t in arange(self.Nt):
            volume = float32(vid.getVolume(t)[::ds, ::ds, :])
            locs = self.particleSetGrouped[t][:, :3]
            radii, Ibg, Ipeak, SNR = estimateRadii((locs, volume))
            self.particleSetGrouped[t][:, 4] = radii
            self.particleSetGrouped[t][:, 5] = Ibg
            self.particleSetGrouped[t][:, 6] = Ipeak
            self.particleSetGrouped[t][:, 7] = SNR
    def estimateRadii(self, trackingData, vidFile, ds=1):
        print('estimating PSF radii')
        self._estimateAllRadii(vidFile, nprocs, ds)
        r, Ibg, Ipeak, SNR = [], [], [], []
        for k, v in self.particleSetGrouped.iteritems():
            r.extend(v[:, 4])
            Ibg.extend(v[:, 5])
            Ipeak.extend(v[:, 6])
            SNR.extend(v[:, 7])
        trackingData.setDetections(
            self.particleSet
                .assign(r=array(r))
                .assign(Ibg=array(Ibg))
                .assign(Ipeak=array(Ipeak))
                .assign(SNR=array(SNR))
            )
        return self
