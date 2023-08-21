function out = centerLatentInitialization(scaling, rotation, translation, img)

translationMAT = zeros(3); translationMAT(3, 1:2) = translation;
tmat = [cosd(rotation), -sind(rotation), 0;...
    sind(rotation), cosd(rotation), 0;...
    0 0 1] * ...
    diag([scaling(1), scaling(2), 1]) + ...
    translationMAT;
invTMAT = eye(3);
invmat = inv(tmat);
invTMAT(:,1:2) = invmat(:,1:2);
tform = affine2d(invTMAT);


centerOutput = affineOutputView(size(img),tform,'BoundsStyle','centerOutput');
out.map = imwarp(img, tform,'OutputView',centerOutput);
out.Transformation = tmat;

end

