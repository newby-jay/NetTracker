from numpy import *
import os
import tensorflow as tf
import pandas as pd
from scipy import interpolate
from numpy.random import rand
from itertools import product, cycle

def segmentVid_V2(vid, stats, modelFileName):
    """Process video through a neural network to extract particle centers.
    Uses version 2 neural network, which estimates 2-point conditional
    probabilities. Only works for 2D images."""
    Nt, Ny, Nx = vid.shape
    # P = zeros((Nt, Ny, Nx), dtype='float32')
    # P = zeros((5, 0))
    localizations = []
    def getImage(k):
        vmean, vstd = stats[k]
        img = vid[k].astype('float32')
        inds = img == 0
        # vstdNow = np.std(np.float64(vid[k]))
        img = (img - vmean)/vstd
        img[inds] = 0
        return img.reshape(1, Ny, Nx, 1)
    # def normalize(phi):
    #     e1 = np.exp(phi[..., 1] - phi[..., 0])
    #     Z = e1 + 1.
    #     return e1/Z
    # def extractLikelihood(p):
    #     P0 = P.reshape(Ny, Nx, 52)
    #     Px = normalize(P0[..., :2])
    #     Pxy = normalize(
    #         P[..., 2:].reshape(Ny, Nx, 25, 2)
    #         ).reshape(Ny, Nx, 5, 5)
    #     return Px, Pxy
    with tf.Graph().as_default():
        saver = tf.train.import_meta_graph(modelFileName+'.meta')
        #saver = tf.train.Saver(max_to_keep=None)
        sess = tf.InteractiveSession()
        saver.restore(sess, modelFileName)
        images_placeholder = tf.get_collection('eval_op')[0]
        state_placeholder = tf.get_collection('eval_op')[1]
        PxOp = tf.get_collection('eval_op')[2]
        PxyOp = tf.get_collection('eval_op')[3]
        # Pop = tf.get_collection('eval_op')[2]
        toNextFrame = tf.get_collection('eval_op')[4]
        # Pop0 = tf.get_collection('eval_op')[4]
        # toNextFrame0 = tf.get_collection('eval_op')[5]
        # state0 = zeros((1, int(Ny/2), int(Nx/2), 6), dtype='float32')
        # fd = {images_placeholder: getImage(0),
        #       state_placeholder: state0}
        # P0, toNextFrame0Eval = sess.run([Pop0, toNextFrame0], feed_dict = fd)
        #### Forward run
        state = zeros((1, int(Ny/2), int(Nx/2), 6), 'float32')
        for t in arange(Nt):
            # state = toNextFrameEval if t>0 else toNextFrame0Eval
            fd = {images_placeholder: getImage(t),
                  state_placeholder: state}
            Px, Pxy, toNextFrameEval = sess.run(
                [PxOp, PxyOp, toNextFrame],
                feed_dict = fd)
            state = toNextFrameEval
            #Px, Pxy = extractLikelihood(predictions)
            OL = _ObsL(
                Px.reshape(Ny, Nx),
                Pxy.reshape(Ny, Nx, 5, 5))
            locs = OL.collectLocalizations()
            toAppend = []
            for x in locs:
                z = 0.
                toAppend.append([x[0], x[1], z, t, x[2]])
            localizations.extend(toAppend)
        sess.close()
    return float32(localizations) ## x, y, z, t, p

class _ObsL:
    def __init__(self, Px, Pxy, w=4):
        self.shape = Px.shape
        self._Px = Px
        self._PX = zeros_like(Px)
        self.jointLikelihood = Px.copy()
        self.Pxy = Pxy
        self.mask = zeros(self.shape, int64)
        self.Y = []
        self.window = arange(-w, w+1)
    @staticmethod
    def BLinterpolatePoint(r, A):
        x, y = r
        xa, xb = floor(r[0]), ceil(r[0])
        ya, yb = floor(r[1]), ceil(r[1])
        P00 = A[int(ya), int(xa)]
        P10 = A[int(ya), int(xb)]
        P01 = A[int(yb), int(xa)]
        P11 = A[int(yb), int(xb)]
        if xa == xb and ya == yb:
            return P00
        if xa == xb:
            return P00 + (y - ya)/(yb - ya)*(P11 - P00)
        if ya == yb:
            return P00 + (x - xa)/(xb - xa)*(P11 - P00)
        Q = matrix([[P00, P01], [P10, P11]])
        w = matrix([xb - x, x - xa])
        u = matrix([[yb - y], [y - ya]])
        ret = w*Q*u/(xb - xa)/(yb - ya)
        return ret[0, 0]
    @staticmethod
    def BLpatch(A):
        assert all(A.shape == (5, 5))
        patch = zeros((9, 9))
        patch[::2, ::2] = A.copy()
        P00 = A[:-1, :-1]
        P10 = A[:-1, 1:]
        P01 = A[1:, :-1]
        P11 = A[1:, 1:]
        patch[1::2, 1::2] = (P00 + P01 + P10 + P11)/4.
        patch[1::2, ::2]  = 0.5*(A[:-1, :] + A[1:, :])
        patch[::2, 1::2] = 0.5*(A[:, :-1] + A[:, 1:])
        return patch
    def conditionalPatch(self, y):
        assert y.size == 2
        y1, y2 = int64(around(y))
        patch0 = self.Pxy[y2, y1]
        patch = self.BLpatch(patch0)
        return patch
    def update(self, Y):
        self.Y.extend(Y)
        Ny, Nx = self.shape
        R = 4
        L = 2*R + 1
        for y in Y:
            y1, y2 = int64(around(y))
            patch = self.conditionalPatch(y)
            i0, i1 = max(0, y2-R), min(Ny, y2+R+1)
            j0, j1 = max(0, y1-R), min(Nx, y1+R+1)
            k0, k1 = max(R-y2, 0), min(L, L + Ny - (R+1+y2))
            l0, l1 = max(R-y1, 0), min(L, L + Nx - (R+1+y1))
            self.mask[i0:i1, j0:j1] += 1
            self._PX[i0:i1, j0:j1] += patch[k0:k1, l0:l1]
    def update_joint_likelihood(self):
        binaryMask = self.mask > 0
        PX = (1 - binaryMask)*self._Px + binaryMask*self._PX
        PX[binaryMask] /= self.mask[binaryMask]**2
        self.jointLikelihood = PX
    def reset(self):
        self._PX = zeros_like(self._Px)
        self.Y = []
    def localmax(self, x0):
        "Compute an approximation of local max given nearby node x0"
        ### This needs to be implemented
        ### cannot use bilinear interpolation
        PX = self.jointLikelihood
        Ny, Nx = self.shape
        j0, i0 = x0
        nn = product(self.window, self.window)
        inds = [(i+i0, j+j0) for i, j in nn if 0 <= i+i0 < Ny and 0 <= j+j0 < Nx]
        assert len(inds) > 0
        x = array([
            array([j, i])*PX[i, j] for i, j in inds]).sum(axis=0)
        U = array([PX[i, j] for i, j in inds]).sum()
        assert U > 0
        x /= U
        return x
    def conditionalMLpoint(self):
        PX = self.jointLikelihood
        r = unravel_index(PX.argmax(), self.shape)
        x = array([r[1], r[0]])
        px = PX[r[0], r[1]]
        return x, px
    def collectLocalizations(self):
        locs = []
        n = 0
        while True:
            x0, px = self.conditionalMLpoint()
            if px < 0.5:
                return locs
            x = self.localmax(x0)
            locs.append([x[0], x[1], px])
            self.update([x0])
            self.update_joint_likelihood()
            n += 1
            assert n < 10**4, 'to avoid infinite loop'
