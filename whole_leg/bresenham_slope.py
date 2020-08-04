import numpy as np


def _bres_nslope(slope):
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bress(start, end, miter):

    if miter == -1:
        miter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = +_bres_nslope(end - start)

    stepseq = np.arange(1, miter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    return np.array(np.rint(bline), dtype=start.dtype)


def bres(start, end, miter=5):
    end = np.asarray(end)
    start = np.asarray(start)
    return _bress(start, end, miter).reshape(-1, start.shape[-1])
