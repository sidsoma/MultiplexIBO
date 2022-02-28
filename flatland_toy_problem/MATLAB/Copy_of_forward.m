function [observations, hists] = forward(las_x, las_y, pulseWidth, numSpots, det_x, ...
                                            det_y, numPixels, visibility, numBins, t, plotDat)
    %% constant variables
    c = 3E8;
    
    %% compute pathlengths
    lens = zeros(numPixels, numSpots);
    for i=1:numPixels
        for j=1:numSpots
            x1 = las_x(j); y1 = las_y(j); x2 = det_x(i); y2 = det_y(i); 
            pathLen = (x1^2+y1^2)^0.5 + ((x2-x1)^2+(y2-y1))^0.5 + (x2^2+y2^2)^0.5;
            lens(i, j) = pathLen;
        end
    end
    
    %% compute histograms
    observations = zeros(numPixels, numBins);
    hists = zeros(numPixels, numSpots, numBins);
    for i = 1:numPixels
        for j=1:numSpots
            start = lens(i, j) / c;
            a = rectangularPulse(start, start+pulseWidth, t);
            hists(i, j, :) = a;
            if visibility(i, j) == 1
                observations(i, :) = observations(i, :) + a;
            end
        end
    end
    
    %% plot data
    if plotDat
        %% plot source and illumination
        figure; plot(las_x, las_y, 'or');
        hold on;
        plot(det_x, det_y, 'ob');
        plot(0, 0, 'og')
        hold off;
        legend('laser spot', 'detector', 'iPhone');
        xlim([-2, 2]);
        ylim([-0.5, 2]);
        
        %% plot simulated histograms
        figure;
        for i = 1:numPixels
            a = observations(i, :);
            plot(t * 1E9, a);
            hold on;
            xlabel('time (ns)');
            title(i);
        %     for j = 1:numSpots
        %         plot(t*1E9, squeeze(hists(i, j, :)));
        %     end
            hold off;
            pause;
        end
    end
end