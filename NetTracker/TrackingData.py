from numpy import *
import pandas as pd
from itertools import product, permutations, repeat
from scipy.io import savemat, loadmat
# from numba import jit
import sys
if sys.version_info.major == 3:
    izip = zip
    imap = map
else:
    from itertools import izip, imap

class TrackingData:
    """Process, store, and serialize particle tracking data."""

    def __init__(self, filename='', shape=(), zscale=1):
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
