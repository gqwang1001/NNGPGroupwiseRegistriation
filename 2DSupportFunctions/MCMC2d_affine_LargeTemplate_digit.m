function out = MCMC2d_affine_LargeTemplate_digit(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = int32(dataIn.nsubj);
ncoord = length(dataIn.coord);
ncoord_l = length(dataIn.coord0);
coord = dataIn.coord;
coord_tmp = dataIn.coord0;
uniq_crd_x = unique(coord(:,1));
incre = uniq_crd_x(2) - uniq_crd_x(1);
parworkers = min(16, nsubj);
npar = 6;

Xlatent = init.XL(:);
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
% nnX = dataIn.nnX;
nnX_L = dataIn.nnX_L;
% nnDict = dataIn.nnDict;
nnDict_L = dataIn.nnDict_L;

% t0 = [init.scaling; init.shift]';
% coordTransf = t0 * [coord, ones(ncoord,1)]';
t0 = init.tVec;
Yvec = [];
for subj = 1:nsubj
    coordTransf = Transform_coord_affine(init.tMat(:,:,subj), coord);
    nnIDXs(subj) = nns_2d_square(nnDict_L, coord_tmp, coordTransf, incre, K);
    Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord_tmp, nnIDXs(subj), rho, K, 1e-10);
    
    Ytemp = Y(:,:,subj);
    Yvec(:,subj) = Ytemp(:);
end

tMat = zeros(niter,nsubj,size(t0,2));
tMat_exp = tMat;
bMat = zeros(niter,nsubj);
XMat = zeros(niter+1, ncoord_l);
XMat(1,:) = Xlatent;

sigmaMat = zeros(niter, nsubj);
alphaMat = zeros(niter, 1);
rhoMat = zeros(niter, 1);
acceptMat = zeros(niter, nsubj, npar);
acceptMatRho = zeros(niter, 1);
% bi = ones(nsubj,1);

% XTs = zeros(ncoord, nsubj);

s_beta = hyperPars.s_beta;
mu_beta= hyperPars.mu_beta;
sa=hyperPars.sa;
sb=hyperPars.sb;
priors = hyperPars.priors;
% l0 = hyperPars.l0;

ropt = hyperPars.ropt;
% rhat = hyperPars.rhat;
adapt = hyperPars.adapt;
k=hyperPars.k;
c0=hyperPars.c0;
c1=hyperPars.c1;
logsig = hyperPars.logsig;
Sig = hyperPars.Sig;
logsigRho = hyperPars.logsigRho;
bnds_rho = hyperPars.bnds_rho;
DTs = zeros(nsubj, 6);
Xlatent_cut = cutTemplate(Xlatent, coord, coord_tmp);

for iter = 1:niter
    
    [DT_log, DT_exp] = updatesDerivative(Y, Xlatent_cut, b, t0, exp(hyperPars.logsig)/2, iter, floor(niter/5), k);
    
    % update transformation
    tupdate1 = transfUpdate_2d_largeX_affine_parallel_mex(t0, DT_exp, nnIDXs, Xlatent, Y, sigma, alpha, rho,...
        coord,coord_tmp, K, b, nnDict_L, logsig, Sig, Cov_NN, incre, nsubj, hyperPars.bnds(1:3,:), priors,...
        parworkers, hyperPars.jointUpdates, hyperPars.ta0, hyperPars.tb0, hyperPars.l0);

    tupdate = CenterTransforms_largeX_A(tupdate1, Y, nnIDXs, nsubj, coord, coord_tmp, incre, K, nnDict_L, false);
    
    %     Y_inv = tupdate.Y_inv;
    acceptMat(iter,:, :) = tupdate1.accept;
    t0 = tupdate.transf_exp;
    tMat(iter,:,:) = tupdate.transf_log;
    tMat_exp(iter,:,:) = tupdate.transf_exp;
    nnIDXs = tupdate.nnIDXs;
    
    if iter > floor(niter/5)
        ropt = 0.234;
    end
    
    % adaptive MCMC
    [logsig, Sig] = adaptiveMCMC(adapt, iter, k, tstart, tMat, acceptMat, ...
        logsig, Sig, c0, c1, ropt, nsubj, npar);
    
    % update X(T) and bi, sigma
    [Cov_NN, XTs, bi, sigma] = update_XT_bi_sigma_mex(Xlatent, ncoord, coord_tmp, nnIDXs,...
        rho, K, nsubj, Cov_NN, Yvec, alpha, sigma, b,mu_beta, s_beta, sa, sb, parworkers);
    
    b = exp(log(bi)-mean(log(bi)))';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    % update alpha
    muFsX = CondLatentX_2d_mex(Xlatent, coord_tmp, rho, nnX_L, false, 1e-9);
    alpha = alphaUpdate_parallel_mex(Xlatent, XTs, nsubj, Cov_NN, muFsX, 1e-9, 1e-9, parworkers);
    alphaMat(iter) = alpha;
    
    % update latent map X
    Xlatent = Xupdate_2d_parallel_mex(XTs, Xlatent, nnIDXs, nnX_L, nsubj, alpha,...
        Cov_NN, coord_tmp, rho, parworkers);
    XMat(iter+1,:) = Xlatent';
    Xlatent_cut = cutTemplate(Xlatent, coord, coord_tmp);

    % update rho
    rho = RhoUpdate_2d_parallel_mex_v1(Xlatent, Yvec, b, sigma, rho, coord_tmp, ...
        nnIDXs, nsubj, K, alpha, bnds_rho,...
        exp(logsigRho), Cov_NN, muFsX, nnX_L, parworkers);
    rhoMat(iter) = rho.rho;
    Cov_NN = rho.Covs_nn_t;
    acceptMatRho(iter) = rho.accept;
    rho = rho.rho;
    logsigRho = adaptiveMCMC_rho(adapt, iter, k, acceptMatRho, c0, c1, ropt, logsigRho);
    if rem(iter,k)==0
        disp(iter);
        disp(toc(tstart));
        showOnefig(Xlatent, 1,iter)
        drawnow
        showSummaryFigs(XMat, 2, iter, k)
        drawnow
        showOnefig(Xlatent_cut, 3, iter)
        drawnow
        showTransf(tMat_exp, 4, iter)
        drawnow
    end
end

out.Transf = tMat;
out.Transf_exp = tMat_exp;
out.b = bMat;
out.X = XMat;
out.sigma = sigmaMat;
out.alpha = alphaMat;
out.rho = rhoMat;

out.time = toc(tstart);
out.init = init;
out.dataIn = dataIn;
out.nIter = niter;
out.hyperPars = hyperPars;
end



