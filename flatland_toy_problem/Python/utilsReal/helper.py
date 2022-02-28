import numpy as np
import scipy.io as sio
from scipy.signal import chirp, find_peaks, peak_widths
from scipy import signal
import matplotlib.pyplot as plt

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

def lowPassFilter(t, a, bin_width, w_thresh, plotResults):
    # compute fft
    a_fft = np.fft.fft(a)
    a_fft_shift = np.fft.fftshift(a_fft)
    f = np.fft.fftfreq(len(t), bin_width)
    f_shift = np.fft.fftshift(f)
    
    if plotResults:
        plt.figure()
        plt.title('Magnitude')
        plt.plot(f_shift, np.abs(a_fft_shift))
    
    # compute original reconstruction
    reconst = np.real(np.fft.ifft(np.fft.fftshift(a_fft_shift)))
    if plotResults:
        plt.figure()
        plt.title('Reconstruction (Original)')
        plt.plot(t, reconst)
        plt.plot(t, a)
        print('Original: ' + str(np.sum(np.abs(reconst-a))/len(reconst)))
        
    # compute lowpass reconstruction
    fft_filtered = a_fft_shift * (np.abs(f_shift) < w_thresh)
    if plotResults:
        plt.figure()
        plt.title('Filtered Magnitude')
        plt.plot(f_shift, np.abs(fft_filtered))
    
    rec_filt = np.real(np.fft.ifft(np.fft.fftshift(fft_filtered)))
    if plotResults:
        plt.figure()
        plt.title('Reconstruction (Filtered)')
        i1 = 0; i2 = -1
        plt.plot(t[i1:i2], a[i1:i2])
        plt.plot(t[i1:i2], rec_filt[i1:i2])
        print('Filter: '+str(np.sum(np.abs(rec_filt-a))/len(rec_filt)))
    return rec_filt

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


def computeVisibility3D(det_locs, las_locs, y_min, y_max, z_min, z_max):
    # input:
    #    det_locs: numPixels_z x numPixels_y x 3
    #    las_locs: numSpots_z x numSpots_y x 3
    # output:
    #    vis_gt: numPixels_z x numPixels_y x numSpots
            
    numPixels_z, numPixels_y = det_locs.shape[0:2]
    numSpots = las_locs.shape[1]
    
    det_x = np.tile(np.reshape(det_locs[:, :, 0], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    det_y = np.tile(np.reshape(det_locs[:, :, 1], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    det_z = np.tile(np.reshape(det_locs[:, :, 2], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    
    las_x = np.tile(np.reshape(las_locs[:, :, 0], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    las_y = np.tile(np.reshape(las_locs[:, :, 1], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    las_z = np.tile(np.reshape(las_locs[:, :, 2], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    
    mx = det_x - las_x
    my = det_y - las_y
    mz = det_z - las_z
    
    t = -las_x / mx
    
    y_int = las_y + t * my
    z_int = las_z + t * mz
    
    mask_y = ((y_int > y_max) | (y_int < y_min))
    mask_z = ((z_int > z_max) | (z_int < z_min))
    
    vis = (mask_y | mask_z).astype(int)
    return vis 
        