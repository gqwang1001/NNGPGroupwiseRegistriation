function out = sym_MCMC2d(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = int32(dataIn.nsubj);
ncoord = length(dataIn.coord);
coord = dataIn.coord;
incre = coord(2) - coord(1);
parworkers = min(16, nsubj);
npar = 6;

Xlatent = init.X;
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
nnX = dataIn.nnX;
nnDict = dataIn.nnDict;
% t0 = [init.scaling; init.shift]';
% coordTransf = t0 * [coord, ones(ncoord,1)]';
t0 = init.tVec;
Yvec = [];
for subj = 1:nsubj
    coordTransf = Transform_coord_affine(init.tLogVec(subj, :), coord);
    nnIDXs(subj) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
    Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
    
    Ytemp = Y(:,:,subj);
    Yvec(:,subj) = Ytemp(:);
end

tMat = zeros(niter,nsubj,size(t0,2));
tMat_exp = tMat;
bMat = zeros(niter,nsubj);
XMat = zeros(niter, ncoord);
sigmaMat = zeros(niter, nsubj);
alphaMat = zeros(niter, 1);
rhoMat = zeros(niter, 1);
acceptMat = zeros(niter, nsubj, npar);
acceptMatRho = zeros(niter, 1);
bi = ones(nsubj,1);

XTs = zeros(ncoord, nsubj);

s_beta = hyperPars.s_beta;
mu_beta= hyperPars.mu_beta;
sa=hyperPars.sa;
sb=hyperPars.sb;
priors = hyperPars.priors;

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

for iter = 1:niter
    DTs = updatesDerivative(Y, Xlatent, b, t0, exp(hyperPars.logsig), iter, floor(niter/5));
    
    % update transformation
    tupdate1 = sym_transfUpdate_2d_parfor_mex(t0, DTs, nnIDXs, Xlatent, Y, sigma,...
        alpha, rho, coord, K, b, nnDict, logsig, Sig, Cov_NN, incre, nsubj, ...
        hyperPars.ta0, hyperPars.tb0, priors,...
        hyperPars.l0, parworkers, hyperPars.jointUpdates, hyperPars.bnds);
    
    tupdate = CenterTransforms(tupdate1, Y, nnIDXs, nsubj,coord, incre, K, nnDict, true);
    
    Y_inv = tupdate.Y_inv;
    t0 = tupdate.transf_exp;
    tMat(iter,:,:) = tupdate.transf_log;
    tMat_exp(iter,:,:) = tupdate.transf_exp;
    acceptMat(iter,:, :) = tupdate1.accept;
    nnIDXs = tupdate.nnIDXs;
    
    if iter > floor(niter/5)
        ropt = 0.234;
    end
    
    % adaptive MCMC
    [logsig, Sig] = adaptiveMCMC(adapt, iter, k, tstart, tMat, acceptMat, ...
        logsig, Sig, c0, c1, ropt, nsubj, npar);
    
    % update X(T) and bi, sigma
    [Cov_NN, XTs, bi, sigma] = sym_update_XT_bi_sigma_mex(Y_inv, Xlatent, ...
        ncoord, coord, nnIDXs, rho, K, nsubj, Cov_NN, ...
        Y, alpha, sigma, b, mu_beta, s_beta, sa, sb, parworkers);
    b = exp(log(bi)-mean(log(bi)))';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    % update alpha
    muFsX = CondLatentX_2d_mex(Xlatent, coord, rho, nnX);
    alpha = alphaUpdate_parallel_mex(Xlatent, XTs, nsubj, Cov_NN, muFsX, 1e-9, 1e-9, parworkers);
    alphaMat(iter) = alpha;
    
    % update latent map X
    Xlatent = sym_Xupdate_2d_parallel_mex(Y_inv, b, sigma, XTs, Xlatent,...
        nnIDXs, nnX, nsubj, alpha, Cov_NN, coord, rho, parworkers);
    XMat(iter,:) = Xlatent';
    
    % update rho
    rho = RhoUpdate_2d_parallel_mex_v1(Xlatent, Yvec, b, sigma, rho, coord, ...
        nnIDXs, nsubj, K, alpha, bnds_rho,...
        exp(logsigRho), Cov_NN, muFsX, nnX, parworkers);
    rhoMat(iter) = rho.rho;
    Cov_NN = rho.Covs_nn_t;
    acceptMatRho(iter) = rho.accept;
    rho = rho.rho;
    logsigRho = adaptiveMCMC_rho(adapt, iter, k, acceptMatRho, c0, c1, ropt, logsigRho);
    
    if rem(iter,k)==0
        showOnefig(Xlatent, 1)
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



