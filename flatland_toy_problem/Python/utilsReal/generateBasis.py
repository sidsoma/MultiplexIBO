import numpy as np
import torch
import math

def dot(vec1, vec2):
    # input: m x n x 3 matrices
    return np.sum(vec1 * vec2, axis=-1)

def generateBasisFunction(las_locs, det_locs, t, baseline, pulseShape, bin_width, c):
    # inputs:
    #     src_locs - (x, y) pixels (numPixels x 3)
    #     det_locs - (x, y) pixels (numSpots x 3)
    #
    # output: 
    #     hists    - only pixels of interest (numPixels x numSpots x numBins)
    
    # constant parameters
    numPixels_y, numPixels_x = det_locs.shape[0:2]
    numSpots_y, numSpots_x = las_locs.shape[0:2]
    numBins = len(t)
    
    hists = np.zeros((numSpots_y, numSpots_x, numPixels_y, numPixels_x, numBins))
    tofs = np.zeros((numSpots_y, numSpots_x, numPixels_y, numPixels_x))
    for y in range(numSpots_y):
        for x in range(numSpots_x):
            detLocs = det_locs
            lasLoc = np.tile(np.reshape(las_locs[y, x, :], (1, 1, 3)), (numPixels_y, numPixels_x, 1))
            s = np.tile(np.reshape(baseline, (1, 1, 3)), (numPixels_y, numPixels_x, 1))

            # compute tof
            r1 = np.linalg.norm(lasLoc - s, axis=2)
            r2 = np.linalg.norm(detLocs - lasLoc, axis=2)
            r3 = np.linalg.norm(detLocs, axis=2) 
            pathLen = r1 + r2 + r3
            tof = (pathLen / c) - 4.3E-10
            tofs[y, x, :, :] = tof

            # create array of pulses in fourier domain
            pulse_fft = np.reshape(np.fft.fft(np.squeeze(pulseShape)), (1, 1, numBins)) # 1 x 1 x numBins
            pulses_fft = np.tile(pulse_fft, (numPixels_y, numPixels_x, 1)) # numPixels x x numSpots x numBins
            f = np.reshape(np.fft.fftfreq(numBins, bin_width), (1, 1, numBins)) # 1 x 1 x numBins
            f_tiled = np.tile(f, (numPixels_y, numPixels_x, 1)) # numPixels x numSpots x numBins

            # shift pulses by tof
            tof_r = np.reshape(tof, (numPixels_y, numPixels_x, 1))
            freq_shift = pulses_fft * np.exp(-1j*2*math.pi*f_tiled*tof_r) 
            hists[y, x, :, :, :] = np.abs(np.fft.ifft(freq_shift)) # assume pulse is completely positive. if not, use np.real()
    
#             # attenuation by albedo, r^2 falloff, cosine falloff (NOTE: make sure albedo values are reasonable)
#             r2_squared = r2**2
#             r3_squared = r3**2

#             rho1 = np.tile(np.reshape(albedo[las_yidx, las_xidx], (1, numSpots)), (numPixels, 1))
#             rho2 = np.tile(np.reshape(albedo[det_yidx, det_xidx], (numPixels, 1)), (1, numSpots))

#             n1 = np.tile(np.reshape(n_vector[las_yidx, las_xidx, :], (1, numSpots, 3)), (numPixels, 1, 1))
#             n2 = np.tile(np.reshape(n_vector[det_yidx, det_xidx, :], (numPixels, 1, 3)), (1, numSpots, 1))

#             wc1_unnorm = np.tile(np.reshape(pt_loc[las_yidx, las_xidx, :], (1, numSpots, 3)), (numPixels, 1, 1))
#             wc1 = wc1_unnorm / np.reshape(np.sum(wc1_unnorm**2, 2)**0.5, (numPixels, numSpots, 1))
#             w1c = -wc1
#             wc2_unnorm = np.tile(np.reshape(pt_loc[det_yidx, det_xidx, :], (numPixels, 1, 3)), (1, numSpots, 1))
#             wc2 = wc2_unnorm / np.reshape(np.sum(wc2_unnorm**2, 2)**0.5, (numPixels, numSpots, 1))
#             w12_unnorm = wc2_unnorm - wc1_unnorm
#             w12 = w12_unnorm / np.reshape(np.sum(w12_unnorm**2, 2)**0.5, (numPixels, numSpots, 1))
#             w21 = -w12

#             nc = np.tile(np.reshape(nc, (1, 1, 3)), (numPixels, numSpots, 1))
#             a = rho1 * rho2 * (dot(w12, n1) * dot(w21, n2) * dot(wc2, nc) / r2_squared)
#         #     a = (rho1 * dot(w1c, n1) / r2_squared) * (rho2 * dot(w21, n2) / r3_squared) * dot(wc2, nc)

#             hists *= scale * np.reshape(a, (numPixels, numSpots, 1)) 
    
    return tof, hists
