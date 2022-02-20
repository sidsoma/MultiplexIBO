clc; clear all; close all;

errors = [];
for i = 2:15
    %% user-defined parameters
    % constant parameters
    c = 3E8;
    
    % laser parameters
    pulseWidth = 1E-9; % must be larger than detector resolution
    numSpots = i;
    las_x = linspace(0.5, 1.5, numSpots);
    las_y = 2-las_x; % illumination wall 
    
    % detector parameters
    detectorRes = 300E-12; 
    numPixels = 10;
    det_x = linspace(-1.5, -0.5, numPixels);
    det_y = 2+det_x; % detector wall
    
    % miscellaneous parameters
    maxDist = 3 * (max(det_x)^2 + max(det_y))^2;
    visibility = rand(numPixels, numSpots) < 0.5;
    numBins = ceil(((maxDist+2)/c) / detectorRes);
    t = linspace(0, detectorRes * numBins, numBins);
    plotDat = 0;
    
    %% calculate histograms
    [observations, hists] = forward(las_x, las_y, pulseWidth, numSpots, det_x, det_y, ...
                                        numPixels, visibility, numBins, t, plotDat);
    
    v_reconst = inverse(observations, hists, numPixels, numSpots, numBins);
    
    %% compare ground truth
    error = sum(abs(v_reconst(:)-visibility(:)));
    disp([num2str(i), ' spots --> Error: ', num2str(error)]);
    errors = [errors, error];

%%
plot(errors);
xlabel('Number of Spots');
ylabel('Error in Visibility Matrix');