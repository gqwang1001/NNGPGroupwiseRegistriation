function out = createAffineTransformedImages(scaling, rotation, translation, img)

translationMAT = zeros(3); translationMAT(3, 1:2) = translation;
tmat = [cosd(rotation), sind(rotation), 0;...
       -sind(rotation), cosd(rotation), 0;...
       0 0 1] * ...
       diag([scaling(1), scaling(2), 1]) + ...
       translationMAT;
   
tform = affine2d(tmat);
centerOutput = affineOutputView(size(img),tform,'BoundsStyle','CenterOutput');
out = imwarp(img, tform,'OutputView',centerOutput);    

end

