function WKY = warpRBF_1d(coord, coord_transf, KY)

nc = size(KY, 2);
WKY = KY;

for i=1:nc
    WKY(:,i) = interp1(coord, KY(:,i), coord_transf,'spline');
end

end