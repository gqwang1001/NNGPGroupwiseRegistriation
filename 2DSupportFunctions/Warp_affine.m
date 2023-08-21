function out = Warp_affine(tvec_log, img, logIndicator)

% translationMAT = zeros(3); translationMAT(3, 1:2) = translation;
% tmat = [cosd(rotation), -sind(rotation), 0;...
%     sind(rotation), cosd(rotation), 0;...
%     0 0 1] * ...
%     diag([scaling(1), scaling(2), 1]) + ...
%     translationMAT;
% invTMAT = eye(3);
% invmat = inv(tmat);
% invTMAT(:,1:2) = invmat(:,1:2);
if logIndicator==true
    tmat_exp = expm(Vec2Mat(tvec_log, true));
else
    tmat_exp = Vec2Mat(tvec_log, false);
end

% invtmat = inv(expm(tmat));
% invtmat(:,3) = [0 0 1];
% tform = affine2d(invtmat);

% centerOutput = affineOutputView(size(img),tform,'BoundsStyle','centerOutput');
% Outputform = affineOutputView(size(img),tform,'BoundsStyle','sameAsInput');
% out = imwarp(img, tform,'OutputView',Outputform);
[nx, ny] = size(img);
[coord_x, coord_y] = meshgrid(1:nx, 1:ny);
coords_transformed = [coord_x(:), coord_y(:), ones(nx*ny,1)] * tmat_exp;
out = interp2(coord_x, coord_y,...
    img, ...
    reshape(coords_transformed(:,1), [nx,ny]),...
    reshape(coords_transformed(:,2), [nx,ny]), ...
    "cubic", 0);
end

