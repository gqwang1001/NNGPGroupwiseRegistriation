function [DT_log, DT_exp, inv_DT_log, inv_DT_exp] = sym_updatesDerivative(Y, Latent, b, Tinit, invTinit, stepsize, iter, iterThresh, k )
% This function approximates the


nsubj = length(b);
% DT = zeros(nsubj, 6);
DT_log = zeros(3,3,nsubj);
inv_DT_log = zeros(3,3,nsubj);

DT_exp = repmat(eye(3), [1 1 nsubj]);
inv_DT_exp = repmat(eye(3), [1 1 nsubj]);

if iter<iterThresh && rem(iter, k)==0
    Latentmap = zeros(size(Y(:,:,1)));
    Latentmap(:) = Latent;
    for i = 1:nsubj
        Tinit_i = Vec2Mat(Tinit(i,:), false);
%         Reg = OneStepDerivative(Y(:,:,i), b(i)*Latentmap, Tinit_i, stepsize(i, 1));
        Reg = registerImagesInitialize_affine(Y(:,:,i), b(i)*Latentmap, Tinit_i, 1e-5, 1e-2, 10);
        DT_exp(:,:,i) = Reg.Transformation.T/Tinit_i; % T*inv(Ti)
        DT_log(:,:,i) = logm(DT_exp(:,:,i));
        
        invTinit_i = Vec2Mat(invTinit(i,:), false);
%         invReg = OneStepDerivative(b(i)*Latentmap, Y(:,:,i),  invTinit_i, stepsize(i, 1));
%         inv_DT_exp(:,:,i) = invReg.Transformation.T / invTinit_i;

        invReg = registerImagesInitialize_affine( b(i)*Latentmap,Y(:,:,i), (invTinit_i), 1e-5, 1e-2, 10);
        inv_DT_exp(:,:,i) = invReg.Transformation.T / invTinit_i;

        inv_DT_log(:,:,i) = logm(inv_DT_exp(:,:,i));
    end
end
end

function [MOVINGREG] = OneStepDerivative(MOVING, FIXED, Tinit, stepsize)

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
[xFixed,yFixed] = meshgrid(1:size(FIXED,2),1:size(FIXED,1));
[xMoving,yMoving] = meshgrid(1:size(MOVING,2),1:size(MOVING,1));
sumFixedIntensity = sum(FIXED(:));
sumMovingIntensity = sum(MOVING(:));
fixedXCOM = (fixedRefObj.PixelExtentInWorldX .* (sum(xFixed(:).*double(FIXED(:))) ./ sumFixedIntensity)) + fixedRefObj.XWorldLimits(1);
fixedYCOM = (fixedRefObj.PixelExtentInWorldY .* (sum(yFixed(:).*double(FIXED(:))) ./ sumFixedIntensity)) + fixedRefObj.YWorldLimits(1);
movingXCOM = (movingRefObj.PixelExtentInWorldX .* (sum(xMoving(:).*double(MOVING(:))) ./ sumMovingIntensity)) + movingRefObj.XWorldLimits(1);
movingYCOM = (movingRefObj.PixelExtentInWorldY .* (sum(yMoving(:).*double(MOVING(:))) ./ sumMovingIntensity)) + movingRefObj.YWorldLimits(1);
translationX = fixedXCOM - movingXCOM;
translationY = fixedYCOM - movingYCOM;

% Coarse alignment
initTform = affine2d();
if sum(Tinit, 'all') == 0
    initTform.T(3,1:2) = [translationX, translationY];
else
    initTform.T(:, 1:2) = Tinit(:,1:2);
end

tform = imregtform(MOVING,movingRefObj,FIXED,fixedRefObj,'affine',optimizer,metric,'PyramidLevels',3,'InitialTransformation',initTform);
MOVINGREG.Transformation = tform;

end

