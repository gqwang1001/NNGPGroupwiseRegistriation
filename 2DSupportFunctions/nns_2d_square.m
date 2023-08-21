function out = nns_2d_square(nnDict, coord, coordTrans, incre, K)
% coord: coordinates of template images
% coordTrans: coordinates of moving images

[minco, maxco] = bounds(nnDict.newcoords);

ImgSz = [(maxco(1)-minco(1))/incre+1, (maxco(2)-minco(2))/incre+1]; %image size of the dictionary grid

n_coord = size(coordTrans, 1);
nnIdx = zeros(n_coord, K);
nnDists = zeros(n_coord, K);
% 
% edges_idx = nnDict.edges_idx;
% edge_coords = nnDict.edge_coords;

round_coordTrans = round(coordTrans);

% cut the grid into 9 regions
% #1|#2|#3
% #4|#5|#6
% #7|#8|#9

for i = 1:n_coord
    
    if round_coordTrans(i,1) < minco(1)      
        if round_coordTrans(i,2) < minco(2) % #1
            idx = sub2ind(ImgSz, 1, 1);
        elseif round_coordTrans(i,2) <= maxco(2) % #2
            idx = sub2ind(ImgSz, 1, floor((round_coordTrans(i,2)-minco(2))/incre)+1);
        else % #3
            idx = sub2ind(ImgSz, 1, floor((maxco(2)-minco(2))/incre)+1);
        end
    elseif round_coordTrans(i,1) < maxco(1)
        if round_coordTrans(i,2) < minco(2) % #4
            idx = sub2ind(ImgSz, floor((round_coordTrans(i,1)-minco(1))/incre)+1, 1);
        elseif round_coordTrans(i,2) <= maxco(2) % #5
            idx = sub2ind(ImgSz, floor((round_coordTrans(i,1)-minco(1))/incre)+1, floor((round_coordTrans(i,2)-minco(2))/incre)+1);
        else % #6
            idx = sub2ind(ImgSz, floor((round_coordTrans(i,1)-minco(1))/incre)+1, floor((maxco(2)-minco(2))/incre)+1);
        end
    else
        if round_coordTrans(i,2) < minco(2) % #7
            idx = sub2ind(ImgSz, floor((maxco(1)-minco(1))/incre)+1, 1);
        elseif round_coordTrans(i,2) <= maxco(2) % #8
            idx = sub2ind(ImgSz, floor((maxco(1)-minco(1))/incre)+1, floor((round_coordTrans(i,2)-minco(2))/incre)+1);
        else % #9
            idx = sub2ind(ImgSz, floor((maxco(1)-minco(1))/incre)+1, floor((maxco(2)-minco(2))/incre)+1);
        end
    end
    nnIdx(i,:) = nnDict.idx(idx,:);
    nnDists(i,:) = sqrt(sum(([coordTrans(i,1).*ones(K,1), coordTrans(i,2).*ones(K,1)] - coord(nnIdx(i,:),:)).^2, 2));
end
out.Idxs = nnIdx;
out.Dists = nnDists;
end