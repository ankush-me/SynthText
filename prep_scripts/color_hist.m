




%===============================================
% from the paper:
% SuperParsing: Scalable Nonparametric Image
% Parsing with Superpixels ECCV2010


function [ desc ] = color_hist( im, mask, varargin )

assert(isa(im, 'uint8'));


desc = [];
numBins = 11;
binSize = 256/numBins;
binCenters = (binSize-1)/2:binSize:255;
for c = 1:3
    r = im(:,:,c);
    desc = [desc hist(double(r(mask>0)),binCenters)/sum(mask(:)>0)];
end

end



