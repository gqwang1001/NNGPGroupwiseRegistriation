function [KY, KC] = rbf(coords, KN, shape)
[centersx, centersy] = meshgrid((min(coords(:))-10): KN: (max(coords(:))+10));
centers = [centersx(:), centersy(:)];
KC = exp(-squareform(pdist(centers)).^2/2/(shape^2));
KY = exp(-pdist2(coords, centers).^2/2/(shape^2));
end

