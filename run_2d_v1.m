%%
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\");
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\MCMC\supportFunctions\");
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\MCMC\2DSupportFunctions\");
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\MCMC\RBF");

%% load MINST data
[XTrain,YTrain] = digitTrain4DArrayData;

%% Prepare data
Imgs = [];
imgsInput = [];

for i = 1:3
    temp = XTrain(:,:,:,i+3);
    Imgs(:,:,i) = temp;
    imgsInput(:,i) = temp(:);
end
L = 5;
[crds, crds_temp] = CreateCoordinates(temp, L);

dat = struct();
dat.XInput = Imgs;
dat.coord = crds;
dat.coord_template = crds_temp;

dat.Nsubj = size(imgsInput, 2);

coords = [dat.coord, ones(length(dat.coord(:,1)),1)];
Asig = kron(eye(3),inv(coords'*coords));

figure; 
for i =1:3
    subplot(1,3, i);
    imagesc(Imgs(:,:,i)); colormap jet; colorbar;
end


%% initialization with iterative steps
rng(1);
initials = initialization_affine(Imgs, L, 2, 500,1e-2);
figure; imagesc(initials.Latent); colormap jet; colorbar;
figure; imagesc(initials.Latent_enlarged); colormap jet; colorbar;

%% set up initial parameters and hyperparameters
dataIn.Y = dat.XInput;
dataIn.coord = dat.coord;
dataIn.coord0 = dat.coord_template;
dataIn.nsubj = int32(3);
dataIn.K1 = 5;
dataIn.K2 = 10;
dataIn.K = 9;
dataIn.nnX = nnsearch_2d_v1(dataIn.coord , dataIn.K);
dataIn.nnX_L = nnsearch_2d_v1(dataIn.coord0, dataIn.K);
dataIn.nnDict = nnDictionary_2d(dataIn.coord, 20, dataIn.K);
dataIn.nnDict_L = nnDictionary_2d(dataIn.coord0, 10, dataIn.K);

init.b = initials.b';
init.X = initials.Latent(:);
init.XL = initials.Latent_enlarged(:);

init.tLogVec = initials.tVeclog;
init.tVec = initials.tVec;

init.tVecInv= initials.invtVec;
init.tMat = initials.Tinit;

init.sigma = 1e-4*ones(1, dataIn.nsubj);
init.alpha = .5;
init.rho = .5;
bnds = [0.5,1.5;-45,45;-15,15;.5, 2]; %scaling, rotation, shifting
bnds = ones(6,2);
bnds(:,1) = -bnds(:,2);
bnds(3,:) = [-20, 20];
bnds(6,:) = [-20, 20];
niter = 10e4;
% hyperPars
hyperPars = struct;
hyperPars.s_beta = 1;
hyperPars.mu_beta= 1;
hyperPars.bi = zeros(dataIn.nsubj,1);
hyperPars.sa=1e-5;
hyperPars.sb=1e-5;
hyperPars.l0= (Asig);
hyperPars.ta0 = 1e-5;
hyperPars.tb0 = 1e-5;
hyperPars.priors = repmat([1 0 0 0 1 0], 3,1);
hyperPars.invpriors = repmat([1 0 0 0 1 0], 3,1);

hyperPars.nx = 28;
hyperPars.ny = 28;
hyperPars.ropt = 0.234;
hyperPars.rhat = zeros(dataIn.nsubj, 6);
hyperPars.adapt = true;
hyperPars.k=100;
hyperPars.c0=1;
hyperPars.c1=0.8;
hyperPars.bnds = bnds;
hyperPars.L = L;
hyperPars.KN =2;
hyperPars.shape = 1.5;

hyperPars.lambda_sym = [10 10];

logsig1 = 2.4^2*1e-3;
hyperPars.logsig = repmat(log(logsig1), dataIn.nsubj, 6);
hyperPars.Sig = repmat(diag(diag(1e2*Asig(1:6,1:6))/logsig1),[1 1 dataIn.nsubj]);

invlogsig1 = 2.4^2*1e-3;
hyperPars.invlogsig = repmat(log(invlogsig1), dataIn.nsubj, 6);
hyperPars.invSig = repmat(diag(diag(1e2*Asig(1:6,1:6))/invlogsig1),[1 1 dataIn.nsubj]);

hyperPars.jointUpdates = false;

hyperPars.logsigRho = log(2.4^2/2);
hyperPars.bnds_rho = [.1, 3];
%%
rng('default');
out2d = sym_MCMC2d_affine_LargeTemplate(init, dataIn, niter, hyperPars);
% save D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\results/sym_Large_NNGP_2d_fit_digit7_withDet.mat out2d -v7.3;

%%

ranges = niter-(1:5e4);
LatentImg = zeros(38, 38);
LatentImg(:) = mean(out2d.X(ranges, :));
sdImg = ones(size(LatentImg));
sdImg(:) = std(out2d.X(ranges, :), 0, 1);

figure('position', [100,100,3e3/2, 1e3/3]);
subplot(1,3, 1);imagesc(LatentImg); colormap jet; colorbar; title('Posterior Mean');
subplot(1,3, 2);imagesc(sdImg);  colormap jet; colorbar;title('Posterior SD'); caxis([0, 1]);
subplot(1,3, 3);imagesc(LatentImg./sdImg); colormap jet; colorbar;title('Posterior T-stat');

%% warped images
dat = dataIn.Y;
ranges = niter-(1:5e4);

% Transf_mean = squeeze(mean(out2d.Transf_exp(ranges,:,:), 1));
Transf_mean = squeeze(mean(out2d.invtMat_exp(ranges,:,:), 1));

WarpedImgs = dat;
fig = figure('position', [0,300,4e3/2, 1e3/3]);
for i = 1:3
subplot(1,4,i);
tMat1 = Vec2Mat(Transf_mean(i,:), false);
% WarpedImgs(:,:,i) = Warp_affine_mat(inv(tMat1),dat(:,:,i));
WarpedImgs(:,:,i) = Warp_affine_mat((tMat1),dat(:,:,i));
imagesc(WarpedImgs(:,:,i)); 
colormap jet; colorbar; title(i);
end
subplot(1,4, 4);imagesc(mean(WarpedImgs, 3)); colormap jet; colorbar; title('Mean');%caxis([cmin/2, cmax/2]);


figure('position', [100,100,3e3/2, 1e3/3]);
subplot(1,3, 1);imagesc(mean(WarpedImgs, 3)); colormap jet; colorbar; title('Mean');%caxis([cmin/2, cmax/2]);
subplot(1,3, 2);imagesc(std(WarpedImgs,0, 3)/sqrt(3)); colormap jet; colorbar; title('SE');%caxis([0 2e-4]);
subplot(1,3, 3);imagesc(mean(WarpedImgs, 3)./(std(WarpedImgs,0, 3)/sqrt(3))); colormap jet; colorbar;title('T-stat');caxis([0 20]);

%% summarize the latent images

LatentImg = zeros(38, 38);
LatentImg(:) = mean(out2d.X(ranges, :));
sdImg = ones(size(LatentImg));
sdImg(:) = std(out2d.X(ranges, :), 0, 1);

LatentImg_Cut = zeros(28);
sdImg_Cut = zeros(28);
LatentImg_Cut(:) = cutTemplate(LatentImg(:), crds, crds_temp);
sdImg_Cut(:) = cutTemplate(sdImg(:), crds, crds_temp);

figure('position', [100,100,3e3/2, 1e3/3]);
subplot(1,3, 1);imagesc(LatentImg_Cut); colormap jet; colorbar; title('Posterior Mean');
subplot(1,3, 2);imagesc(sdImg_Cut);  colormap jet; colorbar;title('Posterior SD'); %caxis([0.03, 0.13]);
subplot(1,3, 3);imagesc(LatentImg_Cut./sdImg_Cut); colormap jet; colorbar;title('Posterior T-stat');
