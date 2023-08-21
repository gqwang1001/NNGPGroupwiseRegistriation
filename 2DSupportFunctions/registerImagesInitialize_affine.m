function [MOVINGREG] = registerImagesInitialize_affine(MOVING,FIXED, Tinit,minstepLength, maxstepLength, maxiter)

% Default spatial referencing objects
fixedRefObj = imref2d(size(FIXED));
movingRefObj = imref2d(size(MOVING));

% Intensity-based registration
[optimizer, metric] = imregconfig('monomodal');
optimizer.GradientMagnitudeTolerance = 1.00000e-04;
optimizer.MinimumStepLength = minstepLength;
% optimizer.MinimumStepLength = 1e-5/2;
optimizer.MaximumStepLength = maxstepLength;
% optimizer.MaximumStepLength = 1.25000e-04;
optimizer.MaximumIterations = maxiter;
% optimizer.MaximumIterations = 500;
optimizer.RelaxationFactor = 0.500000;

% % Align centers
% [xFixed,yFixed] = meshgrid(1:size(FIXED,2),1:size(FIXED,1));
% [xMoving,yMoving] = meshgrid(1:size(MOVING,2),1:size(MOVING,1));
% sumFixedIntensity = sum(FIXED(:));
% sumMovingIntensity = sum(MOVING(:));
% fixedXCOM = (fixedRefObj.PixelExtentInWorldX .* (sum(xFixed(:).*double(FIXED(:))) ./ sumFixedIntensity)) + fixedRefObj.XWorldLimits(1);
% fixedYCOM = (fixedRefObj.PixelExtentInWorldY .* (sum(yFixed(:).*double(FIXED(:))) ./ sumFixedIntensity)) + fixedRefObj.YWorldLimits(1);
% movingXCOM = (movingRefObj.PixelExtentInWorldX .* (sum(xMoving(:).*double(MOVING(:))) ./ sumMovingIntensity)) + movingRefObj.XWorldLimits(1);
% movingYCOM = (movingRefObj.PixelExtentInWorldY .* (sum(yMoving(:).*double(MOVING(:))) ./ sumMovingIntensity)) + movingRefObj.YWorldLimits(1);
% translationX = fixedXCOM - movingXCOM;
% translationY = fixedYCOM - movingYCOM;

% Align centers
fixedCenterXWorld = mean(fixedRefObj.XWorldLimits);
fixedCenterYWorld = mean(fixedRefObj.YWorldLimits);
movingCenterXWorld = mean(movingRefObj.XWorldLimits);
movingCenterYWorld = mean(movingRefObj.YWorldLimits);
translationX = fixedCenterXWorld - movingCenterXWorld;
translationY = fixedCenterYWorld - movingCenterYWorld;

% Coarse alignment
initTform = affine2d();
if sum(Tinit, 'all') == 0
    initTform.T(3,1:2) = [translationX, translationY];
else
    initTform.T(:,1:2) = Tinit(:, 1:2);
end

% Apply transformation
% tform = imregtform(MOVING,movingRefObj,FIXED,fixedRefObj,'similarity',optimizer,metric,'PyramidLevels',3,'InitialTransformation',initTform);
tform = imregtform(MOVING,movingRefObj,FIXED,fixedRefObj,'affine',optimizer,metric,'PyramidLevels',3,'InitialTransformation',initTform);
MOVINGREG.Transformation = tform;
% invtform = invert(tform);
invtform = imregtform(FIXED,fixedRefObj,MOVING,movingRefObj,'affine',optimizer,metric,'PyramidLevels',3,'InitialTransformation',invert(initTform));
MOVINGREG.InvertTransformation = invtform;
% MOVINGREG.InvRegisteredImage = imwarp(FIXED, fixedRefObj, invtform, 'OutputView', movingRefObj, 'SmoothEdges', true);
% MOVINGREG.RegisteredImage = imwarp(MOVING, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);
sameAsInput = affineOutputView(size(FIXED),tform,'BoundsStyle','SameAsInput');
sameAsInputINV = affineOutputView(size(FIXED),invtform,'BoundsStyle','SameAsInput');
MOVINGREG.InvRegisteredImage = imwarp(FIXED, fixedRefObj, invtform, 'OutputView', sameAsInputINV, 'SmoothEdges', false);
MOVINGREG.RegisteredImage = imwarp(MOVING, movingRefObj, tform, 'OutputView', sameAsInput, 'SmoothEdges', false);

% Store spatial referencing object
MOVINGREG.SpatialRefObj = fixedRefObj;

end

