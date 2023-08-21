function out = MCMC2d_affine(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = int32(dataIn.nsubj);
ncoord = length(dataIn.coord);
coord = dataIn.coord;
uniq_crd_x = unique(coord(:,1));
incre = uniq_crd_x(2) - uniq_crd_x(1);
parworkers = min(16, nsubj);
npar = 6;

Xlatent = init.X(:);
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
nnX = dataIn.nnX;
nnX_L = dataIn.nnX_L;
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
XMat = zeros(niter+1, ncoord);
XMat(1,:) = Xlatent;

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
l0 = hyperPars.l0;

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

for iter = 1:niter
    DTs = updatesDerivative(Y, Xlatent, b, t0, exp(hyperPars.logsig)*1e-5, iter, floor(niter/5));
    
    % update transformation
    tupdate1 = transfUpdate_2d_affine_parallel(t0,DTs, nnIDXs, Xlatent, Y, sigma, alpha, rho,...
        coord, K, b, nnDict, logsig, Sig, Cov_NN, incre, nsubj, hyperPars.bnds(1:3,:), priors,...
        parworkers, hyperPars.jointUpdates, hyperPars.ta0, hyperPars.tb0, hyperPars.l0);
    tupdate = CenterTransforms(tupdate1, Y, nnIDXs, nsubj, coord, incre, K, nnDict, false);
    
    %     Y_inv = tupdate.Y_inv;
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
    [Cov_NN, XTs, bi, sigma] = update_XT_bi_sigma_mex(Xlatent, ncoord, coord, nnIDXs,...
        rho, K, nsubj, Cov_NN, Yvec, alpha, sigma, b,mu_beta, s_beta, sa, sb, parworkers);
    
    b = exp(log(bi)-mean(log(bi)))';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    % update alpha
    muFsX = CondLatentX_2d_mex(Xlatent, coord, rho, nnX, false, 1e-9);
    alpha = alphaUpdate_parallel_mex(Xlatent, XTs, nsubj, Cov_NN, muFsX, 1e-5, 1e-5, parworkers);
    alphaMat(iter) = alpha;
    
    % update latent map X
    Xlatent = Xupdate_2d_parallel_mex(XTs, Xlatent, nnIDXs, nnX, nsubj, alpha,...
        Cov_NN, coord, rho, parworkers);
    XMat(iter+1,:) = Xlatent';
    
    % update rho
    rho = RhoUpdate_2d_parallel_mex(Xlatent, Yvec, b, sigma, rho, coord, ...
        nnIDXs, nsubj, K, alpha, bnds_rho,...
        exp(logsigRho), Cov_NN, muFsX, nnX, parworkers);
    rhoMat(iter) = rho.rho;
    Cov_NN = rho.Covs_nn_t;
    acceptMatRho(iter) = rho.accept;
    rho = rho.rho;
    logsigRho = adaptiveMCMC_rho(adapt, iter, k, acceptMatRho, c0, c1, ropt, logsigRho);
    if rem(iter,1)==0
        showOnefig(Xlatent, 1,iter)
        drawnow
%         showSummaryFigs(XMat, 2, iter, k)
%         drawnow
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



