from numpy import *
import pandas as pd
from scipy.optimize import minimize_scalar
from itertools import product, permutations, repeat
from lap import lapmod, lapjv
from scipy.io import savemat, loadmat
# from numba import jit
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
    # now = array(hungarian_solve(c))
    _, now = array(lapjv(c, return_cost=False))
    back = arange(now.size)
    linkinds = (back<Nb)*(now<Nn)
    MLlinks = array([back[linkinds], now[linkinds]]).T
    return MLlinks

# @jit(nopython=True)
def connectedComponents(args):
    ## input is a 4xN array (x, y, z, p)
    data, shape = args
    Ny, Nx, Nz = shape
    _, N = data.shape
    # nn = array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
    #             (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
    #             (-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1)])
    nn = array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1)])
    locToPlabel = (data[0]*Ny + data[1])*Nz +  data[2]
    pointLabels = argsort(locToPlabel)
    locToPlabel = sort(locToPlabel)
    segments = zeros(N)#, dtype='int16')
    NsegedStart = 0;
    isOpened = zeros(N)#, dtype='uint8')
    currentSegNumber = 1
    openPoints = [0]
    openPoints.pop()
    localizations = [(0, 0, 0, 0.)]
    localizations.pop()
    while NsegedStart < N:
        ## get starting point to begin new segment
        while NsegedStart < N and segments[NsegedStart] > 0:
            NsegedStart += 1
            ## add first point to the queue
        if NsegedStart < N:
            openPoints.append(NsegedStart)
        ## continue segmenting points at the front of the queue,
        ## while adding unopened nearest neighbor points to the end of the queue
        pointSet = [0]
        pointSet.pop()
        while len(openPoints) > 0:
            point = openPoints.pop()
            pointSet.append(point)
            segments[point] = currentSegNumber
            for n in arange(nn.shape[0]):
                neighbor = data[:3, point] + nn[n]
                NbrInd = (neighbor[0]*Ny + neighbor[1])*Nz + neighbor[2]
                ind = searchsorted(locToPlabel, NbrInd)
                if ind >= locToPlabel.size or locToPlabel[ind] != NbrInd:
                    continue
                else:
                    nextPoint = pointLabels[ind]
                if nextPoint > -1 and isOpened[nextPoint] == 0:
                    isOpened[nextPoint] = 1
                    openPoints.append(nextPoint)
        ## compute average position of the point
        psum = 0.
        xave, yave, zave = 0., 0., 0.
        pmax = 0.
        zvals = [0]
        zvals.pop()
        for point in pointSet:
            psum += data[3, point]
            xave += data[0, point]*data[3, point]
            yave += data[1, point]*data[3, point]
            zave += data[2, point]*data[3, point]
            zvals.append(int(data[2, point]))
            if data[3, point] > pmax:
                pmax = data[3, point]
        ## check if segment is only one z-slice
        #Nslices = unique(zvals).size
        #sliceCheck = Nslices > 1 or Nz == 1
        if psum > 0:
            localizations.append((xave/psum, yave/psum, zave/psum, pmax))
        ## once queue is empty, start a new segment
        currentSegNumber += 1
    return array(localizations)

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

class TrackingData:
    """Process, store, and serialize particle tracking data."""

    def __init__(self, filename='', shape=(), zscale=1):
        self.linkFn = linkFrame
        self._localizer = self._LocateParticlesConnectedComponents
        self.shape = (None, None, None, None)
        self.zscale = zscale
        self.Nt, self.Ny, self.Nx, self.Nz = None, None, None, None
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
    def _LocateParticlesConnectedComponents(self, pixelProb, thresh, nprocs):
        """Given the localization probabilities, compute the set
           of most likely particle locations for each frame."""
        x, y, z, p, times = [], [], [], [], []
        ###########################
        gt = pixelProb.groupby('t')
        def getVolume(t):
            try:
                Pt = gt.get_group(t)
            except KeyError:
                return array([[0, 0, 0, 0]]).T
            return array(Pt[['x', 'y', 'z', 'p']]).T
        probs = imap(getVolume, arange(self.Nt))
        shape = (self.Ny, self.Nx, self.Nz)
        iterable = izip(probs, repeat(shape))
        # with closing(Pool(nprocs)) as pool:
        #     localizer = pool.imap(connectedComponents, iterable, chunksize=2)
        #     for t, sol in enumerate(localizer):
        #         Nnew, _ = sol.shape
        #         times.extend(t*ones(Nnew))
        #         x.extend(sol[:, 0])
        #         y.extend(sol[:, 1])
        #         z.extend(sol[:, 2])
        #         p.extend(sol[:, 3])
        localizer = imap(connectedComponents, iterable)
        for t, sol in enumerate(localizer):
            if sol.size < 4 or len(sol.shape) != 2:
                continue
            Nnew, _ = sol.shape
            times.extend(t*ones(Nnew))
            x.extend(sol[:, 0])
            y.extend(sol[:, 1])
            z.extend(sol[:, 2])
            p.extend(sol[:, 3])
        print(len(p), 'particle localizations found')
        DF = pd.DataFrame(
            array([times, x, y, z, p,
                zeros_like(p),
                zeros_like(p),
                zeros_like(p),
                zeros_like(p)]).T,
            columns=['t', 'x', 'y', 'z', 'p', 'r', 'Ibg', 'Ipeak', 'SNR']
            )
        return DF
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
    def _linkParticles(self, D, nprocs):
        """find most likely links between pixels in adjacent frames.
           Uses linear assignment via the Hungarian algorithm.
        """
        LINKS = dict((int(t), {}) for t in arange(self.Nt))
        def getLocs(t):
            return self.particleSetGrouped[t][:, :3]
        def getProbs(t):
            return self.particleSetGrouped[t][:, 3]
        locs = izip(imap(lambda t: getProbs(t), arange(1, self.Nt)),
                    imap(lambda t: getProbs(t), arange(self.Nt-1)),
                    imap(lambda t: getLocs(t), arange(1, self.Nt)),
                    imap(lambda t: getLocs(t), arange(self.Nt-1)))
        linker = imap(self.linkFn, izip(locs, repeat((D, self.zscale))))
        for t, MLlinks in enumerate(linker):
            if MLlinks.ndim == 2:
                Nlinks, _ = MLlinks.shape if MLlinks.size > 0 else (0, 0)
            else:
                Nlinks = 0
            LINKS[t+1]['Nlinks'] = Nlinks
            LINKS[t+1]['links'] = MLlinks
        skiplocs = izip(imap(lambda t: getProbs(t), arange(2, self.Nt)),
                        imap(lambda t: getProbs(t), arange(self.Nt-2)),
                        imap(lambda t: getLocs(t), arange(2, self.Nt)),
                        imap(lambda t: getLocs(t), arange(self.Nt-2)))
        skiplinker = imap(self.linkFn, zip(skiplocs, repeat((D, self.zscale))))
        for t, MLlinks in enumerate(skiplinker):
            if MLlinks.ndim == 2:
                Nlinks, _ = MLlinks.shape if MLlinks.size > 0 else (0, 0)
            else:
                Nlinks = 0
            LINKS[t+2]['Nskiplinks'] = Nlinks
            LINKS[t+2]['skiplinks'] = MLlinks
        return LINKS
    def _collectTrajectories(self, P, skipLink):
        """Given links, collect all of the absolute positions for each particle.
           This step is just data extraction and formatting; no statistical
           computations happen here."""
        dataOut = []
        pn = 0
        #keep track of what particle positions we've already extracted
        collectedParticles = dict((k, []) for k in arange(self.Nt))
        def pSet(t, n):
            return self.particleSetGrouped[t][n, :3]
        def getRad(t, n):
            return self.particleSetGrouped[t][n, 4]
        def getProb(t, n):
            return self.particleSetGrouped[t][n, 3]
        def getI(t, n):
            return (self.particleSetGrouped[t][n, 5],
                self.particleSetGrouped[t][n, 6],
                self.particleSetGrouped[t][n, 7])
        def loopFn(k, pn, b_in):
            def loopflag(k, b_in):
                if k<self.Nt-1:
                    oneFrame = (P[k]['links'].size>0) \
                        and (b_in in P[k]['links'][:, 0])
                    if skipLink:
                        twoFrame = (P[k+1]['skiplinks'].size>0) \
                            and (b_in in P[k+1]['skiplinks'][:, 0])
                        return oneFrame or twoFrame
                    return oneFrame
                if k==self.Nt-1:
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
                elif skipLink:
                    inds = P[k+1]['skiplinks'][:, 0] == b_in
                    ind = arange(P[k+1]['Nskiplinks'])[inds][0]
                    b, n = P[k+1]['skiplinks'][ind]
                    if n in collectedParticles[k+1]:
                        return
                    xnow, ynow, znow = pSet(k+1, n)
                    radnow = getRad(k+1, n)
                    pnow = getProb(k+1, n)
                    IbgNow, IpeakNow, SNRnow = getI(k+1, n)
                    dataOut.append({'particle': pn, 'frame': k+1,
                                    'x': xnow, 'y': ynow, 'z': znow,
                                    'p': pnow, 'r': radnow,
                                    'Ibg':IbgNow, 'Ipeak':IpeakNow,
                                    'SNR':SNRnow})
                    collectedParticles[k+1].append(n)
                    k += 2
                    b_in = n
        ###############################
        for k in arange(1, self.Nt):
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
    def segmentParticles(self, pixelProb, vidFile=None,
                         thresh=0.5, nprocs=2, ds=1):
        print('identifying probable particles')
        locData = self._localizer(pixelProb, thresh, nprocs)
        if ds>1:
            locData.x *= ds
            locData.y *= ds
        self.setDetections(locData)
        return self
    def estimateRadii(self, vidFile, ds=1, nprocs=2):
        print('estimating PSF radii')
        self._estimateAllRadii(vidFile, nprocs, ds)
        r, Ibg, Ipeak, SNR = [], [], [], []
        for k, v in self.particleSetGrouped.items():
            r.extend(v[:, 4])
            Ibg.extend(v[:, 5])
            Ipeak.extend(v[:, 6])
            SNR.extend(v[:, 7])
        self.setDetections(
            self.particleSet
                .assign(r=array(r))
                .assign(Ibg=array(Ibg))
                .assign(Ipeak=array(Ipeak))
                .assign(SNR=array(SNR))
            )
        return self
    def linkParticles(self, D=10., nprocs=1, skipLink=False):
        print('linking particles')
        P = self._linkParticles(D, nprocs)
        print('collecting trajectories')
        self.Data = self._collectTrajectories(P, skipLink)
        self.trajectoryStats()
        return self
    def _makePSgrouped(self):
        PSG = {}
        groups = self.particleSet.groupby('t')
        for t in arange(self.Nt):
            try:
                g = groups.get_group(t)
                PSG[t] = array(g[['x', 'y', 'z', 'p',
                    'r', 'Ibg', 'Ipeak', 'SNR']])
            except KeyError:
                PSG[t] = array([[], [], [], [], [], [], [], []]).T
        return PSG
    def trajectoryStats(self, output=True):
        """Print information about particle tracks."""
        td = self.Data.groupby('particle')
        self.Nparticles = td.particle.ngroups
        self.lengths = array(td.size())
        if output and self.Nparticles > 0:
            print(self.Nparticles, 'trajectories')
            print('mean trajectory length:', mean(self.lengths),
                  ', std:', std(self.lengths))
    def save(self, filename, withParticleSet=True):
        """Save tracking data to CSV format. If `withParticleSet` is True
        (Default) then localizations are saved to a seperate file."""
        try:
            self.Data.to_csv(filename+' (tracks).csv')
        except:
            pass
        if withParticleSet:
            # np.save(filename, self.particleSet)
            self.particleSet.to_csv(filename+' (localizations).csv')
        return self
    def tracksToDict(self):
        """Convert tracking data from DataFrame format to Python dictionary.
        Output can be used to serialize track data to JSON
        (e.g., pd.json.dumps(output))."""
        dictData = {}
        for p, g in self.Data.groupby('particle'):
            dx, dy, dz = array(diff(g.x)), array(diff(g.y)), array(diff(g.z))
            dt = array(diff(g.frame))
            N = g.x.size
            D2 = sum((dx**2 + dy**2)/(4.*dt*N))
            D3 = sum((dx**2 + dy**2 + self.zscale*dz**2)/(6.*dt*N))
            dictData[p] = {'x': array(g.x),
                           'y': array(g.y),
                           'z': array(g.z),
                           't': array(g.frame),
                           'dx': dx, 'dy': dy, 'dz': dz, 'dt': dt,
                           'p': array(g.p),
                           'r': array(g.r),
                           'Ibg': array(g.Ibg),
                           'Ipeak': array(g.Ipeak),
                           'SNR': array(g['SNR']),
                           'N': N, 'Deff (xy)': D2, 'Deff (xyz)': D3}
        return dictData
    def detectionsToDict(self):
        """Convert localizations data from DataFrame format to Python dictionary.
        Output can be used to serialize track data to JSON
        (e.g., pd.json.dumps(output))."""
        dictData = {'x': array(self.particleSet.x),
                    'y': array(self.particleSet.y),
                    'z': array(self.particleSet.z),
                    't': array(self.particleSet.t),
                    'p': array(self.particleSet.p),
                    'r': array(self.particleSet.r),
                    'Ibg': array(self.particleSet.Ibg),
                    'Ipeak': array(self.particleSet.Ipeak),
                    'SNR': array(self.particleSet['SNR']),
                    'N': self.particleSet.x.size}
        return dictData
    def particleColor(self, p, cmap='jet'):
        """Return a color for track numper `p` for plotting."""
        c = mod(int(p), 256)
        colorMap = matplotlib.pylab.get_cmap(cmap)
        try:
            return colorMap(self._shuffled[c])
        except:
            self._shuffled = np.random.permutation(arange(256))
            return colorMap(self._shuffled[c])
    def filterPathsByLength(self, n):
        "Filter all trajectories that have less than n increments"
        g = self.Data.groupby('particle')
        NparticlesOld = self.Nparticles
        Data = g.filter(lambda x: x.frame.size >= n)
        ## renumber index
        # d = Data.pivot('particle', 'frame').stack()
        # pn = array(d.axes[0].levels[0])
        nl = {}
        n = 0
        for p, g in Data.groupby('particle'):
            nl[p] = n
            n += 1
        Data.particle = Data.particle.apply(lambda x: nl[x])
        ## update data
        Nparticles = Data.particle.max()
        print(NparticlesOld - Nparticles-1, 'particle paths filtered')
        return self.setData(Data)
    def filterParticlesByRadius(self, rCutOff, keep='<'):
        """Filter particle localizations by PSF radius. Filter radius less than
        rCutOff when `keep`='<' (default) and filter radius greater than
        otherwise."""
        print('filtering particles by radius')
        r = self.particleSet.r
        inds = r<rCutOff if keep=='<' else r>rCutOff
        Nfiltered = sum(inds==False)
        self.filteredParticles = self.particleSet[inds].copy()
        PSnew = array(
            self.particleSet[inds][['t', 'x', 'y', 'z', 'p',
                'r', 'Ibg', 'Ipeak', 'SNR']])
        self.particleSet = self.setDetections(
            pd.DataFrame(
                PSnew,
                columns=['t', 'x', 'y', 'z', 'p',
                    'r', 'Ibg', 'Ipeak', 'SNR']
                )
            )
        print(Nfiltered, 'particle localizations filtered')
        return self
    def setDetections(self, detectionData):
        """Data setter for particle localizations (pre linking)."""
        self.particleSet = detectionData
        self.particleSetGrouped = self._makePSgrouped()
    def _getShape(self):
        if self.Nt == None:
            self.Nt = self.Data.frame.max()
        if self.Nx == None:
            self.Nx = self.Data.x.max()
        if self.Ny == None:
            self.Ny = self.Data.y.max()
        if self.Nz == None:
            self.Nz = self.Data.z.max()
        self.shape = (self.Nt, self.Ny, self.Nx, self.Nz)
    def setData(self, data):
        """Data setter for particle tracks (post linking)."""
        self.Data = data.sort_values(['particle', 'frame'], axis=0)
        self.trajectoryStats(output=False)
        self._getShape()
        return self
