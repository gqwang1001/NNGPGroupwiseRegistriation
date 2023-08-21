function nnX = nnsearch(coord, K)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

n = length(coord);
nnIdx = zeros(n, K);
nnDists = zeros(n,K);

nnIdx(1,1) = 1;
 
for i = 2:n
    
    if i<=K
        [nnIdx(i, 1:(i-1)), nnDists(i, 1:(i-1))] = knnsearch(coord(1:(i-1)), coord(i), 'K', K);
    else 
        [nnIdx(i,:), nnDists(i,:)] = knnsearch(coord(1:(i-1)), coord(i), 'K', K);
    end
    
end

nnX.Idxs = nnIdx;
nnX.Dists = nnDists;

end

