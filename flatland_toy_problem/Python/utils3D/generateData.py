import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
from utils_flatland.helper import pulse
import math

def generate3DDataFast(las_locs, det_locs, visibility, numBins, plotData, plotSetup, obj_xmin, obj_xmax, obj_ymin, obj_ymax, usePulse=True, pctNoise=0, jitter=0, pulseWidth=None, t=None):  

    # constant variables
    c = 3E8   
    numSpots = las_locs.shape[1]
    numPixels_z, numPixels_y = det_locs.shape[0:2]
    
    det_x = np.tile(np.reshape(det_locs[:, :, 0], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots)) # numPixels_z x numPixels_y x numSpots
    det_y = np.tile(np.reshape(det_locs[:, :, 1], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    det_z = np.tile(np.reshape(det_locs[:, :, 2], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    
    las_x = np.tile(np.reshape(las_locs[:, :, 0], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1)) # 1 x 1 x numSpots
    las_y = np.tile(np.reshape(las_locs[:, :, 1], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    las_z = np.tile(np.reshape(las_locs[:, :, 2], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    
    # compute tof
    pathLen = (las_x**2 + las_y**2 + las_z**2)**0.5 + ((las_x-det_x)**2 + (las_y-det_y)**2 + (las_z-det_z)**2)**0.5 + (det_x**2+det_y**2+det_z**2)**0.5                   
    tof = np.reshape(pathLen / c, (numPixels_z, numPixels_y, numSpots, 1))
    
    # create array of pulses
    t, a = pulse(0, numBins)
    a_fft = np.fft.fft(a) 
    f = np.reshape(np.fft.fftfreq(len(t), 100E-12), (1, 1, 1, numBins)) # 1 x 1 x 1 x numBins
    a_fft = np.reshape(a_fft, (1, 1, 1, numBins)) # 1 x 1 x 1 x numBins
    pulses_fft = np.tile(a_fft, (numPixels_z, numPixels_y, numSpots, 1)) # numPixels_z x numPixels_y x numSpots x numBins
    
    # shift pulses by t0
    f_tiled = np.tile(f, (numPixels_z, numPixels_y, numSpots, 1)) # numPixels_z x numPixels_y x numSpots x numBins
    freq_shift = pulses_fft * np.exp(-1j * 2*math.pi*f_tiled*tof) 
    hists = np.abs(np.fft.ifft(freq_shift))
    
    # reshape visibility matrix
    visibility = np.reshape(visibility, (numPixels_z, numPixels_y, numSpots, 1))
    
    # add noise and jitter
    noisy_tof = tof + jitter*np.random.normal(0, 1, (numPixels_z, numPixels_y, numSpots, 1))
    trans = pulses_fft * np.exp(-1j * 2*math.pi*f_tiled*noisy_tof)
    jitter_hists = np.real(np.fft.ifft(trans))
    noisy_hists = jitter_hists + (pctNoise * 17500 * np.random.normal(0, 1, (numPixels_z, numPixels_y, numSpots, numBins)))
    observations = np.sum(noisy_hists * visibility, axis=2)

    return torch.tensor(observations), torch.tensor(hists)

def generateDataFast(las_x, las_y, det_x, det_y, visibility, numBins, plotData, plotSetup, obj_xmin, obj_xmax, obj_ymin, obj_ymax, usePulse=True, pctNoise=0, jitter=0, pulseWidth=None, t=None):  
    ############
    # INPUTS: 
    #   visibility = numPixels x numSpots
    # OUTPUTS:
    #   hists = numPixels x numSpots x numBins tensor
    #   observations = numPixels x numBins tensor
    ############
    # constant variables
    c = 3E8   
    numSpots = len(las_x)
    numPixels = len(det_x)
    
    det_x = np.tile(np.reshape(det_x, (numPixels, 1)), (1, numSpots))
    det_y = np.tile(np.reshape(det_y, (numPixels, 1)), (1, numSpots))

    las_x = np.tile(np.reshape(las_x, (1, numSpots)), (numPixels, 1))
    las_y = np.tile(np.reshape(las_y, (1, numSpots)), (numPixels, 1))
    
    # compute tof
    pathLen = (las_x**2 + las_y**2)**0.5 + ((las_x-det_x)**2 + (las_y-det_y)**2)**0.5 + (det_x**2+det_y**2)**0.5
    tof = np.reshape(pathLen / c, (numPixels, numSpots, 1))
    
    # create array of pulses
    t, a = pulse(0, numBins)
    a = np.reshape(a, (1, 1, numBins))
    pulseArr = np.tile(a, (numPixels, numSpots, 1))
    
    # shift pulses by t0
    pulses_fft = np.fft.fft(pulseArr)
    f = np.reshape(np.fft.fftfreq(len(t), 100E-12), (1, 1, numBins))
    f = np.tile(f, (numPixels, numSpots, 1))
    trans = pulses_fft * np.exp(-1j * 2*math.pi*f*tof)
    hists = np.real(np.fft.ifft(trans))
    
    visibility = np.reshape(visibility, (numPixels, numSpots, 1))
    
    # add noise and jitter
    noisy_tof = tof + jitter*np.random.normal(0, 1, (numPixels, numSpots, 1))
    trans = pulses_fft * np.exp(-1j * 2*math.pi*f*noisy_tof)
    jitter_hists = np.real(np.fft.ifft(trans))
    noisy_hists = jitter_hists + (pctNoise * 17500 * np.random.normal(0, 1, (numPixels, numSpots, numBins)))
    observations = np.sum(noisy_hists * visibility, axis = 1)

    # plot source and illumination
    if plotSetup:
        plt.figure()
        plt.title('Setup')
        plt.plot(las_x, las_y, 'or')
        plt.plot(det_x, det_y, 'ob')
        plt.plot(0, 0, 'og')
        xMin = min(np.min(las_x), np.min(det_x), 0) - 1
        yMin = min(np.min(las_y), np.min(det_y), 0) - 1
        xMax = max(np.max(las_x), np.max(det_x), 0) + 1
        yMax = max(np.max(las_y), np.max(det_y), 0) + 1
        plt.xlim([xMin, xMax])
        plt.ylim([yMin, yMax])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')     
        if True:
            plt.plot(np.linspace(obj_xmin, obj_xmax, 100), obj_ymin * np.ones(100), 'c')
            plt.plot(np.linspace(obj_xmin, obj_xmax, 100), obj_ymax * np.ones(100), 'c')
            plt.plot(obj_xmin * np.ones(100), np.linspace(obj_ymin, obj_ymax, 100), 'c')
            plt.plot(obj_xmax * np.ones(100), np.linspace(obj_ymin, obj_ymax, 100), 'c')
        if False: 
            plt.plot(-0.25*np.ones(100), np.linspace(0.75, 1.25, 100))
            plt.plot(0.25*np.ones(100), np.linspace(0.75, 1.25, 100))
            plt.plot(np.linspace(-0.25, 0.25, 100), 0.75*np.ones(100))
            plt.plot(np.linspace(-0.25, 0.25, 100), 1.25*np.ones(100))
        plt.legend(['laser', 'detector', 'iPhone', 'object'], loc= 'lower right')
#         colors = ['c', 'm']
#         for i in range(2):
#             for j in range(numPixels):
#                 x1 = las_x[i]; y1 = las_y[i]; x2 = det_x[j]; y2 = det_y[j]
#                 m = (y1 - y2) / (x1 - x2)
#                 x = np.linspace(x1, x2, 100)
#                 y = m * (x - x1) + y1
#                 plt.plot(x, y, '--' + colors[i])

    # plot data
    if plotData:
        # plot multiplexed histograms
        for i in range(numPixels):
            plt.figure()
            a = observations[i, :]
            plt.plot(t * 1E9, a)
            plt.xlabel('time (ns)')
            plt.ylabel('intensity (a.u)')
            plt.title('Observation at pixel ' + str(i+1))
            if False and i == 5:
                plt.savefig('sourceSweep/' + str(numSpots) + 'spots.png')
    return torch.tensor(observations), torch.tensor(hists)