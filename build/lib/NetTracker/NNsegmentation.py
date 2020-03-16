##############################
##############################
#### UNUSED ##################
##############################
##############################
# from __future__ import division
# from __future__ import print_function
import numpy as np
from numpy import array, arange, zeros, ones_like
from numpy import mean, std, r_, c_, nonzero, float64
import os
import tensorflow as tf
import pandas as pd
import NetTracker
from scipy import interpolate
from numpy.random import rand

def segmentVidForwardBackward(vid, stats, modelFileName, backwardRun=False):
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
        state0 = zeros((1, int(Ny/2), int(Nx/2), 6), dtype='float32')
        fd = {images_placeholder: getImage(0),
              state_placeholder: state0}
        P0, toNextFrame0Eval = sess.run([Pop0, toNextFrame0], feed_dict = fd)
        #### Forward run
        for t in arange(Nt):
            state = toNextFrameEval if t>0 else toNextFrame0Eval
            fd = {images_placeholder: getImage(t),
                  state_placeholder: state}
            Pt, toNextFrameEval = sess.run([Pop, toNextFrame], feed_dict = fd)
            P[t] = Pt.reshape(Ny, Nx)
        #### Backward run
        if backwardRun:
            for t in arange(Nt-1)[::-1]:
                fd = {images_placeholder: getImage(t),
                      state_placeholder: toNextFrameEval}
                Pt, toNextFrameEval = sess.run(
                    [Pop, toNextFrame],
                    feed_dict = fd)
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
