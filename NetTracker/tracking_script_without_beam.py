from numpy import *
import NetTrackerEXP as NT
import pims
modelPath = '~/Dropbox/PT/NetTracker/netTrackerRNN/models/3L-RNN-3D-set1-EMA'
vidPath = '~/Dropbox/PT/BeamTest/testVids/'
# vidName = '151020_RAG_D_Duodenum_COOH_4'
# vidName = '151020_RAG_D_Duodenum_PEG_1'
vidName = 'vid1'
fn = os.path.join(vidPath, vidName)
print(fn)
output = {}
with pims.open(fn + '.tif') as frames:
    Nt = len(frames)
    Ny, Nx = frames.frame_shape
    shape = (Nt, Ny, Nx, 1)
    output['videoData'] = array([array(frame) for frame in frames])\
        .reshape(shape)
print(output['videoData'].shape)
vm = output['videoData'].mean(axis=(1, 2, 3))
vs = (float64(output['videoData'])**2).mean(axis=(1, 2, 3))
print(vm.shape, vs.shape)
output['stats'] = array([vm, vs]).T.reshape(Nt, 1, 2)
output['metadata'] = {'fileName': vidName,
                     'chunkIndex': (0, 0, 0, 0),
                     'dt': 1,
                     'dxy': 1,
                     'dz': 1,
                     'vidShape': shape,
                     }
nn = NT.NeuralNet().process(
    ('', output),
    modelPath)
seg = NT.Segment().process(next(nn))
link = next(NT.Linker(filterLength=1).process(next(seg)))
out = link[1]['trackData'][['x', 'y', 'z', 't', 'r', 'Ibg', 'Ipeak', 'SNR', 'particle']]
out.to_csv(os.path.join(csvPath, vidName + '_base.csv'))
