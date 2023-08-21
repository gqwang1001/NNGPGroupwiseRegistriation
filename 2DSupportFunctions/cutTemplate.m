function Xnew = cutTemplate(Xlatent, coord, coordLarge)

nx = length(unique(coord(:,1)));
nx_l = length(unique(coordLarge(:,1)));
ny = length(unique(coord(:,2)));
ny_l = length(unique(coordLarge(:,2)));

Lx = (nx_l-nx)/2;
Ly = (ny_l-ny)/2;

Xmat = reshape(Xlatent, nx_l, []);
Xmat_new = Xmat((1+Lx):(nx+Lx), (1+Ly):(ny+Ly));
Xnew = Xmat_new(:);

end

