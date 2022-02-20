import numpy as np

def carving(visibility, det_locs, las_locs, pixelSize, minX, maxX, eta, zeta, p_e, p_o):
    
    numSpots = las_locs.shape[1]
    numPixels_z, numPixels_y = det_locs.shape[0:2]
    
    det_x = np.tile(np.reshape(det_locs[:, :, 0], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots)) # numPixels_z x numPixels_y x numSpots
    det_y = np.tile(np.reshape(det_locs[:, :, 1], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    det_z = np.tile(np.reshape(det_locs[:, :, 2], (numPixels_z, numPixels_y, 1)), (1, 1, numSpots))
    
    las_x = np.tile(np.reshape(las_locs[:, :, 0], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1)) # 1 x 1 x numSpots
    las_y = np.tile(np.reshape(las_locs[:, :, 1], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    las_z = np.tile(np.reshape(las_locs[:, :, 2], (1, 1, numSpots)), (numPixels_z, numPixels_y, 1))
    
    minY = np.min(np.hstack((det_y, las_y)))
    maxY = np.max(np.hstack((det_y, las_y)))
    
    minZ = np.min(np.hstack((det_z, las_z)))
    maxZ = np.max(np.hstack((det_z, las_z)))
    
    numPixelsX = int(1 + (maxX - minX) / pixelSize)
    numPixelsY = int(1 + (maxY - minY) / pixelSize)
    numPixelsZ = int(1 + (maxZ - minZ) / pixelSize)
    
    xlocs = np.linspace(minX, maxX, numPixelsX)
    ylocs = np.linspace(minY, maxY, numPixelsY)
    zlocs = np.linspace(minZ, maxZ, numPixelsZ)
    
    occupied_prob = np.zeros((numPixelsY, numPixelsX, numPixelsZ))
    unoccupied_prob = np.zeros((numPixelsY, numPixelsX, numPixelsZ))
    naiveMap = np.ones((numPixelsY, numPixelsX, numPixelsZ))
    for i in range(numPixels):
        for j in range(numSpots):
            x_idx = 0
            for x in xlocs:
                x1 = las_x[j]; y1 = las_y[j]; x2 = det_x[i]; y2 = det_y[i]
                m = (y1 - y2) / (x1 - x2)
                y = m * (x - x1) + y1
                k = int(np.ceil((y - minY) / pixelSize)-1)
                if visibility[i, j] == 1:
                    unoccupied_prob[k, x_idx] += 1
                    naiveMap [k, x_idx] = 0
                if visibility[i, j] == 0:
                    occupied_prob[k, x_idx] += 1
                x_idx += 1
    m = unoccupied_prob
    n = occupied_prob
    heatmap = (eta**m * (1-eta)**n * p_o) 
    heatmap /= (1-zeta)**m * zeta**n * p_e + eta**m * (1-eta)**n * p_o
    return xlocs, ylocs, heatmap, naiveMap, occupied_prob, unoccupied_prob

# # naive backprojection and carving code
# def computeShape(visibility, las_x, las_y, det_x, det_y, pixelSize, minX, maxX, method):
#     numSpots = len(las_x)
#     numDets = len(det_x)
    
#     minY = np.min(np.hstack((det_y, las_y)))
#     maxY = np.max(np.hstack((det_y, las_y)))
    
#     numPixelsY = int(1 + (maxY - minY) / pixelSize)
#     numPixelsX = int(1 + (maxX - minX) / pixelSize)
    
#     ylocs = np.linspace(minY, maxY, numPixelsY)
#     xlocs = np.linspace(minX, maxX, numPixelsX)
#     heatmap = np.ones((numPixelsY, numPixelsX))
#     for i in range(numDets):
#         for j in range(numSpots):
#             if method == 'carving' and visibility[i, j] == 1:
#                 x_idx = 0
#                 for x in xlocs:
#                     x1 = las_x[j]; y1 = las_y[j]; x2 = det_x[i]; y2 = det_y[i]
#                     m = (y1 - y2) / (x1 - x2)
#                     y = m * (x - x1) + y1
#                     k = int(np.ceil((y - minY) / pixelSize)-1)
#                     heatmap[k, x_idx] = 0
#                     x_idx += 1
#             if method == 'backprojection' and visibility[i, j] == 0:
#                 x_idx = 0
#                 for x in xlocs:
#                     x1 = las_x[j]; y1 = las_y[j]; x2 = det_x[i]; y2 = det_y[i]
#                     m = (y1 - y2) / (x1 - x2)
#                     y = m * (x - x1) + y1
#                     k = int(np.ceil((y - minY) / pixelSize)-1)
#                     heatmap[k, x_idx] += 1
#                     x_idx += 1
#     return xlocs, ylocs, heatmap