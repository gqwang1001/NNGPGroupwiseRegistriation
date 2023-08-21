function out = centerLatentInitialization_affine(tmat, img)

% translationMAT = zeros(3); translationMAT(3, 1:2) = translation;
% tmat = [cosd(rotation), -sind(rotation), 0;...
%     sind(rotation), cosd(rotation), 0;...
%     0 0 1] * ...
%     diag([scaling(1), scaling(2), 1]) + ...
%     translationMAT;
% invTMAT = eye(3);
% invmat = inv(tmat);
% invTMAT(:,1:2) = invmat(:,1:2);
% 
% invtmat = inv(tmat);
% invtmat(:,3) = [0 0 1];

% tform = affine2d(invtmat);
% OutputFormat = affineOutputView(size(img),tform,'BoundsStyle','sameAsInput');
% out.map = imwarp(img, tform,'OutputView',OutputFormat);

[nx, ny] = size(img);
[coord_x, coord_y] = meshgrid(1:nx, 1:ny);
coords_transformed = [coord_x(:), coord_y(:), ones(nx*ny,1)] * tmat;
out.map = interp2(coord_x, coord_y,...
    img, ...
    reshape(coords_transformed(:,1), [nx,ny]),...
    reshape(coords_transformed(:,2), [nx,ny]), ...
    "cubic", 0);
%  figure; imagesc(out.map);colormap jet;colorbar;
%  
%   figure; imagesc(img);colormap jet;colorbar;

out.Transformation = tmat;

end

