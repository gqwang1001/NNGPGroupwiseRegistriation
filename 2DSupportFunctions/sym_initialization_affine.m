function [out] = sym_initialization_affine(dat, L, initial_subj)

dat(dat<.1e-4) = 0;
nsubj=size(dat,3);
b = ones(nsubj, 1);

if initial_subj == 'all'
    Latent = mean(dat, 3);
else
    Latent = dat(:,:,initial_subj);
end

centers = [0.5+size(dat, 1)/2, 0.5+size(dat, 2)/2];

datRegistered = dat;
datRegistered1 =dat;
Transf = zeros(3,3,nsubj);
% maxiter =1000;
maxiter = 200;

k=0;
eps = 1;

movingReg = cell(nsubj,1);

logT = zeros(3,3,nsubj);
out.logTmat = logT;
out.b = ones(1,nsubj);
Tmat = ones(3,3,nsubj);

while eps>1e-2 && k<maxiter
    k=k+1;
    for i = 1:nsubj
        bi = sum(datRegistered(:,:,i).*Latent, 'all')/sum(Latent(:).^2);        
        if bi>0  &&  bi<10
            b(i)=bi;
        end
        movingReg{i} = registerImagesInitialize_affine(dat(:,:,i), b(i)*Latent, Transf(:,:,i));
        datRegistered(:,:,i) = movingReg{i}.RegisteredImage; %Warped latent map;
        
    end
    b = exp(log(b)-mean(log(b)));
    for i =1:nsubj
        datRegistered1(:,:,i) = datRegistered(:,:,i)/b(i);
    end
    Latent = mean(datRegistered1, 3);
    eps = 0;
    for i=1:nsubj
        eps = eps + mean((datRegistered(:,:,i)-b(i)*Latent).^2, 'all');
    end
    
    % store the transformation
    for i = 1:nsubj
        Tmat(:,:,i) = movingReg{i}.Transformation.T;
        logT(:,:,i) = logm(Tmat(:,:,i));
    end
    
    logT_centered = logT;
    logTVec = zeros(nsubj, 6);
    TVec = zeros(nsubj, 6);
    Tinit = logT;
    mean_logT = mean(logT, 3);
    for i=1:nsubj
        logT_centered(:,:,i) = logT(:,:,i) - mean_logT;
        logTVec(i,:) = [logT_centered(:,1,i)', logT_centered(:,2,i)'];
        Tinit(:,:,i) = expm(logT_centered(:,:,i));
        Tvec(i,1:3) = Tinit(:,1,i); Tvec(i,4:6) = Tinit(:,2,i);
    end

    mean_T = expm(mean_logT);
    shiftCenteredMat = zeros(3);
    %     shiftCenteredMat(3, 1:2) = centers;
    LatentOut = centerLatentInitialization_affine(mean_T-shiftCenteredMat, Latent);
    Latent = LatentOut.map;
  
end

% enlarged latente map
[nx, ny] = size(Latent);
LLmap = zeros(nx+2*L, ny+2*L);
LLmap((L+1):(nx+L), (L+1):(ny+L)) = Latent;


out.b = b;
out.logTmat = logT_centered;
out.tVeclog = logTVec;
out.tVec = Tvec;
out.Tinit = Tinit;
out.b = b;
out.Latent = Latent;
out.Latent_enlarged = LLmap;
out.eps=eps;
end

function out = shiftTransformation(T, center)
shiftMAT = [center, 1]*T;
out = shiftMAT(1:2);
end

function out = shiftTransformationINV(T, center)
shiftMAT = [-center, 1]*T;
out = shiftMAT(1:2);
end

% figure;
% subplot(2,2,1);
% imagesc(movingReg.RegisteredImage); colormap jet;colorbar;
% subplot(2,2,2);
% imagesc(movingReg.InvRegisteredImage); colormap jet;colorbar;
% subplot(2,2,3);
% imagesc(dat(:,:,3)); colormap jet;colorbar;
% subplot(2,2,4);
% imagesc(Latent); colormap jet;colorbar;
%
% figure;
% ids = 12:14;
% nsubjs = length(ids);
% for j = 1:3
%     idx=ids(j);
%     subplot(2,nsubjs, j);
%     imagesc(registeredimage(:,:,idx)); colormap jet;colorbar;
%     subplot(2,nsubjs, j+nsubjs);
%     imagesc(dat(:,:,idx)); colormap jet;colorbar;
% end
%
% figure;
% for j = 1:33
%     subplot(6,6, j);
%     imagesc(datRegistered(:,:,j)); colormap jet;colorbar;
%     title(j);
% end
%
% figure;
% for j = 1:33
%     subplot(6,6, j);
%     imagesc(dat(:,:,j)); colormap jet;colorbar;
%     title(j);
% end

%
%
% figure;imagesc(Latent); colormap jet;colorbar;
% figure;imagesc(out.LatentCentered); colormap jet;colorbar;
%
% figure;
% for j = 1:nsubj
%     subplot(1,nsubj, j);
%     imagesc(registeredimage(:,:,j)); colormap jet;colorbar;
% end

