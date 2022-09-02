import numpy as np
from numpy import array, arange, zeros, ones_like, ones
from numpy import mean, std, r_, c_, nonzero, float64, float32
import os
import tensorflow
import tensorflow.compat.v1 as tf
import pandas as pd
from scipy import interpolate
from numpy.random import rand
from itertools import repeat
import sys
if sys.version_info.major == 3:
    izip = zip
    imap = map
else:
    from itertools import izip, imap


def segmentVid_V1(vid, stats, modelFileName, backwardRun=False):
    """Process video through a neural network to extract particle centers.
    Uses version 2 neural network. Works for 3D videos."""
    Nt, Ny, Nx = vid.shape
    P = zeros((Nt, Ny, Nx), dtype='float32')
    # P = zeros((5, 0))

    def getImage(k):
        vmean, vstd = stats[k]
        img = vid[k].astype('float32')
        inds = img == 0
        # vstdNow = np.std(np.float64(vid[k]))
        img = (img - vmean)/vstd
        img[inds] = 0
        return img.reshape(1, Ny, Nx, 1)
    with tf.Graph().as_default():
        saver = tf.train.import_meta_graph(modelFileName+'.meta')
        #saver = tf.train.Saver(max_to_keep=None)
        sess = tf.InteractiveSession()
        saver.restore(sess, modelFileName)
        images_placeholder = tf.get_collection('eval_op')[0]
        state_placeholder = tf.get_collection('eval_op')[1]
        Pop = tf.get_collection('eval_op')[2]
        toNextFrame = tf.get_collection('eval_op')[3]
        Pop0 = tf.get_collection('eval_op')[4]
        toNextFrame0 = tf.get_collection('eval_op')[5]
        toNextFrameEval = 0
        state0 = zeros((1, int(Ny/2), int(Nx/2), 6), dtype='float32')
        fd = {images_placeholder: getImage(0),
              state_placeholder: state0}
        P0, toNextFrame0Eval = sess.run([Pop0, toNextFrame0], feed_dict=fd)
        #### Forward run
        for t in arange(Nt):
            state = toNextFrameEval if t > 0 else toNextFrame0Eval
            fd = {images_placeholder: getImage(t),
                  state_placeholder: state}
            Pt, toNextFrameEval = sess.run([Pop, toNextFrame], feed_dict=fd)
            P[t] = Pt.reshape(Ny, Nx)
        #### Backward run
        if backwardRun:
            for t in arange(Nt-1)[::-1]:
                fd = {images_placeholder: getImage(t),
                      state_placeholder: toNextFrameEval}
                Pt, toNextFrameEval = sess.run(
                    [Pop, toNextFrame],
                    feed_dict=fd)
                Pt = Pt.reshape(Ny, Nx)
                Q = 1. - P[t]
                P[t] *= Pt
                P[t] /= P[t] + Q*(1. - Pt)
        sess.close()
    inds = P > 0.5
    out0 = pd.DataFrame(array(nonzero(inds)).T, columns=['t', 'y', 'x'])
    dprobOut = array(P[inds].flatten())
    dprobOut[dprobOut > 0.99] = 0.99
    out = out0.assign(p=dprobOut)
    return out


def LocateParticlesConnectedComponents(shape, pixelProb, thresh=0.5):
    """Given the localization probabilities, compute the set
       of most likely particle locations for each frame."""
    Nt, Ny, Nx, Nz = shape
    fshape = (Ny, Nx, Nz)
    x, y, z, p, times = [], [], [], [], []
    ###########################
    gt = pixelProb.groupby('t')

    def getVolume(t):
        try:
            Pt = gt.get_group(t)
        except KeyError:
            return array([[0, 0, 0, 0]]).T
        return array(Pt[['x', 'y', 'z', 'p']]).T
    probs = imap(getVolume, arange(Nt))

    iterable = izip(probs, repeat(fshape))
    localizer = imap(_connectedComponents, iterable)
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
               np.zeros_like(p),
               np.zeros_like(p),
               np.zeros_like(p),
               np.zeros_like(p)]).T,
        columns=['t', 'x', 'y', 'z', 'p', 'r', 'Ibg', 'Ipeak', 'SNR']
        )
    return DF


def _connectedComponents(args):
    ## input is a 4xN array (x, y, z, p)
    data, shape = args
    Ny, Nx, Nz = shape
    _, N = data.shape
    # nn = array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
    #             (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
    #             (-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1)])
    nn = array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                (0, 0, -1), (0, 0, 1)])
    locToPlabel = (data[0]*Ny + data[1])*Nz + data[2]
    pointLabels = np.argsort(locToPlabel)
    locToPlabel = np.sort(locToPlabel)
    segments = zeros(N)  # , dtype='int16')
    NsegedStart = 0
    isOpened = zeros(N)  # , dtype='uint8')
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
                ind = np.searchsorted(locToPlabel, NbrInd)
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
