function [KY, KC] = rbf_1d(coords, KN, shape)
increment = coords(2)-coords(1);
centers = (min(coords(:))-5):(increment*KN):(max(coords(:)+5));
KC = exp(-squareform(pdist(centers')).^2/2/(shape^2));
KY = exp(-pdist2(coords, centers').^2/2/(shape^2));
end
