function out = MCMC2d_rbf(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = int32(dataIn.nsubj);
ncoord = length(dataIn.coord);
coord = dataIn.coord;
coord_tmp = dataIn.coord0;
ncoord_tmp = length(coord_tmp);

incre = coord(2) - coord(1);
parworkers = min(16, nsubj);
npar = 6;

[KY, KC] = rbf(coord_tmp, hyperPars.KN, hyperPars.shape);

Xlatent = init.XL(:);
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
L = hyperPars.L;
WKY = [];
for subj = 1:nsubj
    %     coordTransf = Transform_coord_affine(init.tLogVec(subj, :), coord);
    %     nnIDXs(subj) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
    %     Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
    %
    WKY(:,:,subj)=warpRBF_affine(Vec2Mat(init.tVec(subj,:), false), KY, hyperPars.nx+2*L, hyperPars.ny+2*L, coord_tmp);
    Ytemp = Y(:,:,subj);
    Yvec(:,subj) = Ytemp(:);
end

tMat = zeros(niter,nsubj,size(t0,2));
tMat_exp = tMat;
bMat = zeros(niter,nsubj);
XMat = zeros(niter+1, ncoord_tmp);
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
DT_exp = repmat(eye(3), [1 1 nsubj]);
inv_DT_exp = repmat(eye(3), [1 1 nsubj]);
% w = (KY'*KY)\(KY'*Xlatent);
w = pinv(KY)*Xlatent;

Yexpand = expandMaps(Y, hyperPars.L, nsubj);
YexpandVec = squeeze(reshape(Yexpand, [], 1, nsubj));

% showOnefig(KY*w, 1)
% drawnow
for iter = 1:niter
    
    Xlatent_cut = cutTemplate(Xlatent, coord, coord_tmp);
%     [DT_log, DT_exp] = updatesDerivative(Y, Xlatent_cut, b, t0, exp(hyperPars.logsig)/2, iter, floor(niter/5), k);
    
    % update transformation
    tupdate1 = transfUpdate_2d_rbf_affine_parallel_mex(coord, coord_tmp, t0, DT_exp, Y, WKY, KY, w, sigma,b, logsig,...
        Sig, nsubj, priors, parworkers, false,hyperPars.ta0, hyperPars.tb0, hyperPars.l0,...
        hyperPars.nx +2*L,hyperPars.ny+2*L);
    tupdate = CenterTransforms_rbf(tupdate1, coord_tmp, Y, KY, nsubj, false, hyperPars.nx+2*L,hyperPars.ny+2*L, parworkers);
    WKY = tupdate.WKY;
    %     Y_inv = tupdate.Y_inv;
    t0 = tupdate.transf_exp;
    tMat(iter,:,:) = tupdate.transf_log;
    tMat_exp(iter,:,:) = tupdate.transf_exp;
    acceptMat(iter,:, :) = tupdate1.accept;
    
    if iter > floor(niter/5)
        ropt = 0.234;
    end
    
    % adaptive MCMC
    [logsig, Sig] = adaptiveMCMC(adapt, iter, k, tstart, tMat, acceptMat, ...
        logsig, Sig, c0, c1, ropt, nsubj, npar);
    
    % update X(T) and bi, sigma
    [XTs, bi, sigma] = update_XT_bi_sigma_rbf_mex(coord, coord_tmp, WKY, w, ncoord,...
        nsubj, Yvec, sigma,mu_beta, s_beta, sa, sb, parworkers);
    
    b = exp(log(bi)-mean(log(bi)))';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    w = wUpdates_mex(YexpandVec, WKY, KC, b, parworkers, sigma);
    Xlatent = KY*w;
    XMat(iter+1,:) = Xlatent';
    %     showOnefig(Xlatent, 1)
    %     drawnow
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
% out.alpha = alphaMat;
% out.rho = rhoMat;

out.time = toc(tstart);
out.init = init;
out.dataIn = dataIn;
out.nIter = niter;
out.hyperPars = hyperPars;
end



