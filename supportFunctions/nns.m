function out = nns(nnDict, coord, coordTrans, incre, K)


[minco, maxco] = bounds(nnDict.newcoords);
n_coord = length(coordTrans);

nnIdx = zeros(n_coord, K);
nnDists = zeros(n_coord, K);

for i = 1:n_coord
    
    if coordTrans(i)< minco
        nnIdx(i,:) = nnDict.idx(1,:);
    elseif coordTrans(i) > maxco
        nnIdx(i,:) = nnDict.idx(end,:);
    else
        idx = floor((coordTrans(i)-minco)/incre)+1;
        nnIdx(i,:) = nnDict.idx(idx,:);
    end
    
    nnDists(i,:) = abs(coordTrans(i) - coord(nnIdx(i,:)));
    
end
    out.Idxs = nnIdx;
    out.Dists = nnDists;
end