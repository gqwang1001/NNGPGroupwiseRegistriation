function [coordinates,coordinates_template] = CreateCoordinates(Img, L)
% create coordinates matrices for subject-specific and template Images

[nx, ny] = size(Img);

[crd_x, crd_y] = meshgrid(1:nx, 1:ny);
coordinates = [ crd_x(:), crd_y(:)];

[crd_x_tpl, crd_y_tpl] = meshgrid((1-L):(nx+L), (1-L):(ny+L));
coordinates_template = [ crd_x_tpl(:), crd_y_tpl(:)];

coordinates = coordinates(:,[2,1]);
coordinates_template = coordinates_template(:,[2,1]);
end

