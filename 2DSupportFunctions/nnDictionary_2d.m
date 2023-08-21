function [nnDict] = nnDictionary_2d(coords, l, K)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[minco, maxco] = bounds(coords);
uniq_crd_x = unique(coords(:,1));
incre_x = uniq_crd_x(2)-uniq_crd_x(1);
% incre_y = coords(2,2) - coords(1,2);

% tempMat = ones(round((maxco(1)-minco(1)+2*l+1)/incre_x), round((maxco(2)-minco(2)+2*l+1)/incre_x));
% [newcoordsX, newcoordsY] = find(tempMat==1);
% newcoords = [(newcoordsX-mean(newcoordsX))*incre_x, (newcoordsY-mean(newcoordsY))*incre_x];
% newcoords = [(newcoordsX)*incre_x, (newcoordsY)*incre_x];

[new_x, new_y] = meshgrid((minco(1)-l):incre_x:(maxco(1)+l), (minco(2)-l):incre_x:(maxco(2)+l));
newcoords = [new_y(:), new_x(:)];

nnDict = struct();
[nnDict.idx, nnDict.dist] = knnsearch(coords, newcoords,'K', K);
nnDict.newcoords = newcoords;

[minco_edge, maxco_edge] = bounds(nnDict.newcoords);

edges_idx = find(ismember(nnDict.newcoords(:,1), [minco_edge(1), maxco_edge(1)]) | ...
                      ismember(nnDict.newcoords(:,2), [minco_edge(2), maxco_edge(2)]));
edge_coords = nnDict.newcoords(edges_idx, :);

nnDict.edges_idx = edges_idx';
nnDict.edge_coords = edge_coords;

end


