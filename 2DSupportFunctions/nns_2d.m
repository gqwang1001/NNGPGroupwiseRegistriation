function out = nns_2d(nnDict, coord, coordTrans, incre, K)

[minco, maxco] = bounds(nnDict.newcoords);

ImgSz = [(maxco(1)-minco(1))/incre+1, (maxco(2)-minco(2))/incre+1];

n_coord = size(coordTrans, 1);
nnIdx = zeros(n_coord, K);
nnDists = zeros(n_coord, K);

edges_idx = nnDict.edges_idx;
edge_coords = nnDict.edge_coords;

for i = 1:n_coord
    
    if coordTrans(i,1) >= minco(1) && coordTrans(i,1)<=maxco(1) && ...
       coordTrans(i,2) >= minco(2) && coordTrans(i,2)<=maxco(2)
       idx = sub2ind(ImgSz, floor((coordTrans(i,1)-minco(1))/incre)+1, floor((coordTrans(i,2)-minco(2))/incre)+1);
    else
       idxInEdge = knnsearch(edge_coords, coordTrans(i, :));
       idx = edges_idx(idxInEdge(1));
    end
       nnIdx(i,:) = nnDict.idx(idx,:);
       nnDists(i,:) = sqrt(sum(([coordTrans(i,1).*ones(K,1), coordTrans(i,2).*ones(K,1)] - coord(nnIdx(i,:),:)).^2, 2));
end
    out.Idxs = nnIdx;
    out.Dists = nnDists;
end