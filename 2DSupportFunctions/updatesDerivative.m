function [DT_log, DT_exp] = updatesDerivative(Y, Latent, b, Tinit, stepsize, iter, iterThresh, k )
% This function approximates the

Latent(Latent<0)=0;
Y(Y<0) = 0;

nsubj = length(b);
% DT = zeros(nsubj, 6);
DT_log = zeros(3,3,nsubj);
DT_exp = repmat(eye(3), [1 1 nsubj]);

if iter<iterThresh && rem(iter, k)==0
    
    Latentmap = zeros(size(Y(:,:,1)));
    Latentmap(:) = Latent;
    
    for i = 1:nsubj
        
        Tinit_i = Vec2Mat(Tinit(i,:), false);
        Reg = OneStepDerivative(Y(:,:,i), b(i)*Latentmap, Tinit_i, stepsize(i, 1));
        
        DT_exp(:,:,i) = Reg.Transformation.T * inv(Tinit_i);
%         DT_log = logm(DTMat/stepsize(i, 1));
        DT_log(:,:,i) = logm(DT_exp(:,:,i));
%         DT_log_vec = DT_log(:);
%         DT(i,:) = DT_log_vec(1:6);
    end
    
end


end

function [MOVINGREG] = OneStepDerivative(MOVING,FIXED, Tinit, stepsize)
%registerImages  Register grayscale images using auto-generated code from Registration Estimator app.
%  [MOVINGREG] = registerImages(MOVING,FIXED) Register grayscale images
%  MOVING and FIXED using auto-generated code from the Registration
%  Estimator app. The values for all registration parameters were set
%  interactively in the app and result in the registered image stored in the
%  structure array MOVINGREG.
%
% % Normalize FIXED image
%
% % Get linear indices to finite valued data
% finiteIdx = isfinite(FIXED(:));
%
% % Replace NaN values with 0
% FIXED(isnan(FIXED)) = 0;
%
% % Replace Inf values with 1
% FIXED(FIXED==Inf) = 1;
%
% % Replace -Inf values with 0
% FIXED(FIXED==-Inf) = 0;
%
% % Normalize input data to range in [0,1].
% FIXEDmin = min(FIXED(:));
% FIXEDmax = max(FIXED(:));
% if isequal(FIXEDmax,FIXEDmin)
%     FIXED = 0*FIXED;
% else
%     FIXED(finiteIdx) = (FIXED(finiteIdx) - FIXEDmin) ./ (FIXEDmax - FIXEDmin);
% end
%
% % Normalize MOVING image
%
% % Get linear indices to finite valued data
% finiteIdx = isfinite(MOVING(:));
%
% % Replace NaN values with 0
% MOVING(isnan(MOVING)) = 0;
%
% % Replace Inf values with 1
% MOVING(MOVING==Inf) = 1;
%
% % Replace -Inf values with 0
% MOVING(MOVING==-Inf) = 0;
%
% % Normalize input data to range in [0,1].
% MOVINGmin = min(MOVING(:));
% MOVINGmax = max(MOVING(:));
% if isequal(MOVINGmax,MOVINGmin)
%     MOVING = 0*MOVING;
% else
%     MOVING(finiteIdx) = (MOVING(finiteIdx) - MOVINGmin) ./ (MOVINGmax - MOVINGmin);
% end

% Default spatial referencing objects
fixedRefObj = imref2d(size(FIXED));
movingRefObj = imref2d(size(MOVING));

% Intensity-based registration
[optimizer, metric] = imregconfig('monomodal');
optimizer.GradientMagnitudeTolerance = 1.00000e-04;
% optimizer.MinimumStepLength = 1e-5;
optimizer.MinimumStepLength = stepsize;
optimizer.MaximumStepLength = stepsize;
% optimizer.MaximumStepLength = 6.25000e-02;
% optimizer.MaximumIterations = 100;
optimizer.MaximumIterations = 1;

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
    initTform.T = Tinit;
end

% Apply transformation
% tform = imregtform(MOVING,movingRefObj,FIXED,fixedRefObj,'similarity',optimizer,metric,'PyramidLevels',3,'InitialTransformation',initTform);

tform = imregtform(MOVING,movingRefObj,FIXED,fixedRefObj,'affine',optimizer,metric,'PyramidLevels',3,'InitialTransformation',initTform);

MOVINGREG.Transformation = tform;
% invtform = invert(tform);
% MOVINGREG.InvertTransformation = invtform;
% % MOVINGREG.InvRegisteredImage = imwarp(FIXED, fixedRefObj, invtform, 'OutputView', movingRefObj, 'SmoothEdges', true);
% % MOVINGREG.RegisteredImage = imwarp(MOVING, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);
% 
% 
% sameAsInput = affineOutputView(size(FIXED),tform,'BoundsStyle','SameAsInput');
% sameAsInputINV = affineOutputView(size(FIXED),invtform,'BoundsStyle','SameAsInput');
% MOVINGREG.InvRegisteredImage = imwarp(FIXED, fixedRefObj, invtform, 'OutputView', sameAsInputINV, 'SmoothEdges', false);
% MOVINGREG.RegisteredImage = imwarp(MOVING, movingRefObj, tform, 'OutputView', sameAsInput, 'SmoothEdges', false);
% 
% 
% % Store spatial referencing object
% MOVINGREG.SpatialRefObj = fixedRefObj;

end

