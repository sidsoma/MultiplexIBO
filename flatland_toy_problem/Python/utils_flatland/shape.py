import numpy as np

def carving(visibility, las_x, las_y, det_x, det_y, pixelSize, minX, maxX, eta, zeta, p_e, p_o):
    numSpots = len(las_x)
    numDets = len(det_x)
    
    minY = np.min(np.hstack((det_y, las_y)))
    maxY = np.max(np.hstack((det_y, las_y)))
    
    numPixelsY = int(1 + (maxY - minY) / pixelSize)
    numPixelsX = int(1 + (maxX - minX) / pixelSize)
    
    ylocs = np.linspace(minY, maxY, numPixelsY)
    xlocs = np.linspace(minX, maxX, numPixelsX)
    occupied_prob = np.zeros((numPixelsY, numPixelsX))
    unoccupied_prob = np.zeros((numPixelsY, numPixelsX))
    naiveMap = np.ones((numPixelsY, numPixelsX))
    for i in range(numDets):
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