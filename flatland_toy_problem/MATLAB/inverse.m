function visibility = inverse(observations, hists, numPixels, numSpots, numBins)
    visibility = zeros(numPixels, numSpots);
    for i = 1:numPixels
        y = reshape(observations(i, :), [numBins, 1]);
        A = squeeze(hists(i, :, :)).';
        v = linsolve(A, y);
        visibility(i, :) = v;
    end
    visibility = imbinarize(visibility, 0.5);
end