import numpy as np
import scipy.io as sio
from scipy.signal import chirp, find_peaks, peak_widths
from scipy import signal

############# HELPER FUNCTIONS #####################

def rectangularPulse(t1, t2, t):
    if np.min(t) > t1 or np.max(t) < t2:
        raise Exception('check t bounds')
    return torch.tensor(((t >= t1) * (t <= t2)))

def pulse(t0, numBins):
    a = sio.loadmat('../../pulseShapes_confidential.mat')
    a = a['pulseNP']
    numPad = numBins - len(a)
    numPad = max(numPad, 0)
    a = np.pad(a, (0, numPad), 'edge'); a = a[:, 0]
    t = np.linspace(0, 100E-12 * (numBins-1), numBins)
    a = np.roll(a, round(t0 / 100E-12))
    return t, a

def matchFilt(measureSig, refSig):
    corr = signal.correlate(measureSig, refSig)
    lags = signal.correlation_lags(len(measureSig), len(refSig)) * 1E-10
    if False:
        plt.figure()
        plt.plot(lags, corr)
    peaks, _ = find_peaks(corr)    
    timePeaks = []
    for i in peaks:
        timePeaks.append(lags[i])
    return timePeaks
    
def computeVisibility(las_x, las_y, det_x, det_y, x, y1, y2):
    numPixels = len(det_x)
    numSpots = len(las_x)
    
    det_x = np.tile(np.reshape(det_x, (numPixels, 1)), (1, numSpots))
    det_y = np.tile(np.reshape(det_y, (numPixels, 1)), (1, numSpots))

    las_x = np.tile(np.reshape(las_x, (1, numSpots)), (numPixels, 1))
    las_y = np.tile(np.reshape(las_y, (1, numSpots)), (numPixels, 1))
    
    m = (det_y - las_y) / (det_x - las_x)
    y = m * (x - det_x) + det_y
    
    vis =  1 - ((y >= y1) & (y <= y2))
    return vis 

def computeVisibility2D(las_x, las_y, det_x, det_y, x1, x2, y1, y2, pixelSize):
    numPixels = len(det_x)
    numSpots = len(las_x)
    
    det_x = np.tile(np.reshape(det_x, (numPixels, 1)), (1, numSpots))
    det_y = np.tile(np.reshape(det_y, (numPixels, 1)), (1, numSpots))

    las_x = np.tile(np.reshape(las_x, (1, numSpots)), (numPixels, 1))
    las_y = np.tile(np.reshape(las_y, (1, numSpots)), (numPixels, 1))
    
    x_vals = np.linspace(x1, x2, int((x2-x1)/pixelSize))
    vis = np.ones((numPixels, numSpots))
    for x in x_vals:
        m = (det_y - las_y) / (det_x - las_x)
        y = m * (x - det_x) + det_y
        v =  1 - ((y >= y1) & (y <= y2))
        vis = vis * v
    return vis 
        