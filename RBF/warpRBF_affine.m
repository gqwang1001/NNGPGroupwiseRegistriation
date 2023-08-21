function WKY = warpRBF_affine(deform, KY, nx, ny, coord)

nc = size(KY, 2);
Img = zeros(nx, ny);
WKY = KY;

for i = 1:nc    
    Img(:) = KY(:,i);
    Wimg = Warp_affine_rbf(deform, Img, coord, false);
    WKY(:,i) = Wimg(:);
end

end


