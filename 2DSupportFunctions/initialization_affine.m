function [out] = initialization_affine(dat, L, initial_subj, maxiter, stepLength)

dat(dat<0) = 0;
nsubj=size(dat,3);
b = ones(nsubj, 1);

if initial_subj == 'all'
    Latent = mean(dat, 3);
else
    Latent = dat(:,:,initial_subj);
end

% centers = [0.5+size(dat, 1)/2, 0.5+size(dat, 2)/2];

datRegistered = dat;
datRegistered1 =dat;
Tinit = repmat(zeros(3), [1 1 nsubj]);

k=0;
eps = 1;
epsAll = ones(nsubj, 1);
movingReg = cell(nsubj,1);
logT = zeros(3,3,nsubj);
out.logTmat = logT;
out.b = ones(1,nsubj);
Tmat = ones(3,3,nsubj);

while eps>1e-2 && k<maxiter
    k=k+1;
    if rem(k, 20)==0
        disp(k);
    end

    for i = 1:nsubj
        bi = sum(datRegistered(:,:,i).*Latent, 'all')/sum(Latent(:).^2);

        if bi>0  &&  bi<10
            b(i)=bi;
        end

        movingReg{i} = registerImagesInitialize_affine(dat(:,:,i), b(i)*Latent, Tinit(:,:,i), stepLength, stepLength, 1);
        datRegistered(:,:,i) = movingReg{i}.RegisteredImage;

%         while(epsAll(i)>1e-3)
%             movingReg{i} = registerImagesInitialize_affine(dat(:,:,i), b(i)*Latent, Tinit(:,:,i), stepLength);
%             datRegistered(:,:,i) = movingReg{i}.RegisteredImage;
%             epsAll(i) = mean((datRegistered(:,:,i)-b(i)*Latent).^2, 'all');
%             Tinit(:,:,i) = movingReg{i}.Transformation.T;
%         end

        % store the transformation
        Tmat(:,:,i) = movingReg{i}.Transformation.T;
        logT(:,:,i) = logm(Tmat(:,:,i));
    end

    b = exp(log(b)-mean(log(b)));

    for i =1:nsubj
        datRegistered1(:,:,i) = datRegistered(:,:,i)/b(i);
    end
    Latent = mean(datRegistered1, 3);
    eps = 0;
    for i=1:nsubj
        epsAll(i) =  mean((datRegistered(:,:,i)-b(i)*Latent).^2, 'all');
        eps = eps + epsAll(i);
    end

    centered = init_CenterTransforms_largeX_A(Tmat, Latent);
    Tinit = centered.Tmat_centered;
    Latent = centered.Latent;

end

invLogT = [];
for i = 1:nsubj
    movingReg = registerImagesInitialize_affine(dat(:,:,i), b(i)*Latent, (Tinit(:,:,i)), stepLength/10, stepLength, 10);
    invLogT(:,:,i) = logm(movingReg.InvertTransformation.T);
end
mean_invLogT = mean(invLogT, 3);
invLogT_centered = invLogT - repmat(mean_invLogT, [1,1,nsubj]);
invExpT = [];
invtVec = [];
for i = 1:nsubj
    invExpT(:,:,i) = expm(invLogT_centered(:,:,i));
    invtVec(i, 1:3) = invExpT(:,1,i);
    invtVec(i, 4:6) = invExpT(:,2,i);
end

% enlarged latente map
[nx, ny] = size(Latent);
LLmap = zeros(nx+2*L, ny+2*L);
LLmap((L+1):(nx+L), (L+1):(ny+L)) = Latent;


out.b = b;
% out.logTmat = centered.transf_log;
out.tVeclog = centered.transf_log;
out.tVec = centered.transf_exp;
out.Tinit = Tinit;
out.b = b;
out.Latent = Latent;
out.Latent_enlarged = LLmap;
out.eps=eps;

out.invExpT = invExpT;
out.invLogT = invLogT_centered;
out.invtVec = invtVec;
out.epsAll=epsAll;
end

function out = shiftTransformation(T, center)
shiftMAT = [center, 1]*T;
out = shiftMAT(1:2);
end

function out = shiftTransformationINV(T, center)
shiftMAT = [-center, 1]*T;
out = shiftMAT(1:2);
end


function CenteredOut = init_CenterTransforms_largeX_A(Tmat, Latent)

nsubj = size(Tmat, 3);
t1_log_A = zeros(2,2,nsubj);

for i=1:nsubj
    t1_log_A(:,:,i)=logm(Tmat(1:2,1:2,i));
end

% centerize log T
Tlog_A_mean = mean(t1_log_A, 3);
Tlog_shift_mean = mean(Tmat(3,1:2,:), 3);
Tlog_centered_A = t1_log_A-repmat(Tlog_A_mean, 1,1,nsubj);
Tlog_centered_shift = squeeze(Tmat(3,1:2,:))' - repmat(Tlog_shift_mean, nsubj,1);
Tmean = [expm(Tlog_A_mean), [0;0]; Tlog_shift_mean,1];
% centered latent map
LatentOut = centerLatentInitialization_affine(Tmean, Latent);

Tvec = zeros(nsubj,6);
Tlog_centered = zeros(nsubj,6);
Tmat_centered = zeros(3,3,nsubj);

for i = 1:nsubj
    Texp = [expm(Tlog_centered_A(:,:,i)), [0;0]; Tlog_centered_shift(i,:), 1];
    Tmat_centered(:,:,i) = Texp;
    Tvec(i,:) = [Texp(:,1);Texp(:,2)]';
    Tlog_centered(i,:) = reshape([Tlog_centered_A(:,:,i);Tlog_centered_shift(i,:)], [],1);
end

CenteredOut.transf_log = Tlog_centered;
CenteredOut.transf_exp = Tvec;
CenteredOut.Tmat_centered = Tmat_centered;
CenteredOut.Latent = LatentOut.map;

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

