from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from numpy import *
from numpy.random import rand
import os
import sys
import time
import cStringIO
import zipfile

from NetTracker.TrackingData import TrackingData, estimateRadii
from NetTracker.NNsegmentation import segmentVidForwardBackward
from NetTracker.TrackLinking import linkTracks
segmentVid = segmentVidForwardBackward

import pandas as pd
import tensorflow as tf
from itertools import product

import apache_beam as beam
from apache_beam.transforms import PTransform
from apache_beam.io import filebasedsource, ReadFromText, WriteToText, iobase
from apache_beam.io.iobase import Read

class NeuralNet(beam.DoFn):
    """Process a video with the Neural Net tracker. Input 'backwardRun=False',
    if set to True will use the forward/backward algorithm for better accuracy,
    at the expense of slower processing time."""

    def __init__(self, backwardRun=False):
        self.backwardRun = backwardRun
    def process(self, KVelement, modelPath):
        key, element = KVelement
        stats = mean(element['stats'], axis=1)
        stats[:, 1] = sqrt(stats[:, 1] - stats[:, 0]**2)
        vid = element['videoData']
        Nz = vid.shape[3]
        xyzt = zeros((0, 4), 'int32')
        p = zeros((0), 'float64')
        for z in arange(Nz):
            Pz = segmentVid(vid[..., z], stats,
                            modelPath, self.backwardRun).assign(z=z)
            xyzt = concatenate([xyzt, int32(Pz[['x', 'y', 'z', 't']])], 0)
            pnew = float64(array(Pz['p'])).ravel()
            p = concatenate([p, pnew], 0)
        pointSet = pd.DataFrame(xyzt, columns=['x', 'y', 'z', 't']).assign(p=p)
        output = {'metadata': element['metadata'],
                  'pointSet': pointSet,
                  'stats': stats,
                  'videoData': element['videoData']}
        outputLabel = element['metadata']['fileName']
        nt, ny, nx, nz = element['metadata']['chunkIndex']
        outputLabel += '-{0}-{1}-{2}'.format(nt, ny, nx)
        yield (outputLabel, output)

class Segment(beam.DoFn):
    """Compute particle centers from Neural Network output."""

    def __init__(self):
        pass
    def _getRadii(self, vid, trackData):
        """Estimate radius to local region, limit 15 pixel radius."""
        Nt, Nx, Ny, Nz = trackData.shape
        for t in arange(Nt):
            volume = vid[t]
            assert trackData.particleSetGrouped[t].ndim == 2
            locs = trackData.particleSetGrouped[t][:, :3]
            radii, Ibg, Ipeak, SNR = estimateRadii((locs, volume))
            trackData.particleSetGrouped[int(t)][:, 4] = radii
            trackData.particleSetGrouped[int(t)][:, 5] = Ibg
            trackData.particleSetGrouped[int(t)][:, 6] = Ipeak
            trackData.particleSetGrouped[int(t)][:, 7] = SNR
        r, Ibg, Ipeak, SNR = [], [], [], []
        for k, v in trackData.particleSetGrouped.iteritems():
            r.extend(v[:, 4])
            Ibg.extend(v[:, 5])
            Ipeak.extend(v[:, 6])
            SNR.extend(v[:, 7])
        trackData.setDetections(
            trackData.particleSet
                .assign(r=array(r))
                .assign(Ibg=array(Ibg))
                .assign(Ipeak=array(Ipeak))
                .assign(SNR=array(SNR))
            )
        return trackData
    def process(self, KVelement):
        key, element = KVelement
        Nt, Ny, Nx, Nz = element['videoData'].shape
        trackData = TrackingData(shape=(Nt, Ny, Nx, Nz))
        trackData.segmentParticles(element['pointSet'])
        trackData = self._getRadii(element['videoData'], trackData)
        output = {'pointSet': trackData.particleSet,
                  'metadata': element['metadata']
                 }
        yield (key, output)

class Linker(beam.DoFn):
    """Link particle localization into tracks. Input `sigma=5` determines how
    far a link can be made (increasing this value will mean particles can make
    larger displacements between frames). Input `filterLength=1` (defaults to
    no fitering) filters all particle tracks with fewer than 'filterLength'
    observations. Input 'trackLink=True' set to True will link particles over
    one missing if no link is made to the next frame (this will make tracks
    longer)."""

    def __init__(self, sigma=5., filterLength=5, trackLink=False):
        self.sigma = sigma; self.filterLength = filterLength;
        self.trackLink = trackLink
    def process(self, KVelement):
        key, element = KVelement
        zscale = element['metadata']['dz']/element['metadata']['dxy']
        Nt, Ny, Nx, Nz = element['metadata']['vidShape']
        if Nz > 1:
            l0 = element['pointSet']
            localizations = l0[l0.z % 1 != 0]
        else:
            localizations = element['pointSet']
        trackData = TrackingData(shape=element['metadata']['vidShape'],
                                 zscale=zscale)
        trackData.setDetections(localizations)
        trackData.linkParticles(D=self.sigma)
        detectionsOut = trackData.detectionsToDict()
        trackData.trajectoryStats(output=False)
        if self.trackLink:
            trackData = linkTracks(
                trackData,
                tLinkScale=3,
                birth=2.,
                death=2.)
        if self.filterLength>2:
            trackData.filterPathsByLength(self.filterLength)
        if self.trackLink:
            trackData = linkTracks(
                trackData,
                tLinkScale=15,
                birth=2.,
                death=2.)
        output = {'trackData': trackData.Data,
                  'particleSet': detectionsOut,
                  'tracks': trackData.tracksToDict(),
                  'metadata': element['metadata'],
                 }
        yield (key, output)

# class Serialize(beam.DoFn):
#     """Format localization data for output."""
#
#     def __init__(self):
#         pass
#     def process(self, KVelement):
#         key, element = KVelement
#         zscale = element['metadata']['dz']/element['metadata']['dxy']
#         Nt, Ny, Nx, Nz = element['metadata']['vidShape']
#         if Nz > 1:
#             l0 = element['pointSet']
#             localizations = l0[l0.z % 1 != 0]
#         else:
#             localizations = element['pointSet']
#         trackData = TrackingData(shape=element['metadata']['vidShape'],
#                                  zscale=zscale)
#         trackData.setDetections(localizations)
#         output = {'trackData': '',
#                   'particleSet': trackData.detectionsToDict(),
#                   'tracks': '',
#                   'metadata': element['metadata'],
#                  }
#         yield (key, output)

class TracksToCSV(beam.DoFn):
    """Write tracks to CSV."""

    def __init__(self):
        self.columns = ['x', 'y', 'z', 't', 'r', 'Ibg', 'Ipeak', 'SNR',
                        'Deff (xy)', 'Deff (xyz)', 'particle']
    def _getPath(self, path):
        spath = []
        while len(path) > 0:
            path, name = os.path.split(path)
            spath.append(name)
            if path == '/':
                break
        ## Assuming the AITS inputs have form "gs://bucket/UUID/**.ext"
        assert len(spath) > 3
        spath.reverse()
        assert spath[0] == 'gs:'
        return '/'.join(spath[3:])
    def process(self, KVelement):
        key, element = KVelement
        fileName = self._getPath(key)
        data = zeros((len(self.columns), 0), 'float64')
        Uparticles = sort(array([int(v) for v in element.keys()], 'int'))
        assert all(Uparticles >= 0)
        for pn in Uparticles:
            particle = element[pn]
            particle['particle'] = pn
            assert (particle['x'].size == particle['y'].size
                    == particle['z'].size == particle['t'].size
                    == particle['r'].size)
            N = particle['x'].size
            for c in self.columns:
                Ncheck = array(particle[c]).squeeze().size
                assert Ncheck == 1 or Ncheck == N
            newData = array([particle[c]*ones(N) for c in self.columns])
            data = concatenate([data, newData], 1)
        df = pd.DataFrame(data.T, columns=self.columns)
        yield (fileName + '.csv', df.to_csv(float_format='%.6g', index=False))
