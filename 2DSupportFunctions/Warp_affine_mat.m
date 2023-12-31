function out = Warp_affine_mat(tMat, img)

[nx, ny] = size(img);
[coord_x, coord_y] = meshgrid(1:nx, 1:ny);
coords_transformed = [coord_x(:), coord_y(:), ones(nx*ny,1)] * tMat;
out = interp2(coord_x, coord_y, ...
    img, ...
    reshape(coords_transformed(:,1), [nx,ny]),...
    reshape(coords_transformed(:,2), [nx,ny]), ...
    "cubic", 0);
end

