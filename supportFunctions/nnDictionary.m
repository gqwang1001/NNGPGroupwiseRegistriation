function [nnDict] = nnDictionary(coord,l, K)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


minco = min(coord);
maxco = max(coord);

incre = coord(2)-coord(1);
newcoords = (minco-l):incre:(maxco+l);

nnDict = struct();
[nnDict.idx, nnDict.dist] = knnsearch(coord, newcoords','K', K);
nnDict.newcoords = newcoords';

end

