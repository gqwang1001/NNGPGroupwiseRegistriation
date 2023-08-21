function out = sym_MCMC2d_affine_LargeTemplate(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = int32(dataIn.nsubj);
ncoord = length(dataIn.coord);
ncoord_l = length(dataIn.coord0);

coord_tmp = dataIn.coord0;
L = hyperPars.L;
coord = dataIn.coord;
incre = coord(2) - coord(1);
parworkers = min(16, nsubj);
npar = 6;

Xlatent = init.XL(:);
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
nnX = dataIn.nnX;
% nnDict = dataIn.nnDict;
nnDict_L = dataIn.nnDict_L;
nnX_L = dataIn.nnX_L;

% t0 = [init.scaling; init.shift]';
% coordTransf = t0 * [coord, ones(ncoord,1)]';
t0 = init.tVec;
invt0 = init.tVecInv;

Yvec = [];
for subj = 1:nsubj
    coordTransf = Transform_coord_affine(init.tLogVec(subj, :), coord);
    nnIDXs(subj) = nns_2d_square(nnDict_L, coord_tmp, coordTransf, incre, K);
    Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord_tmp, nnIDXs(subj), rho, K, 1e-10);

    Ytemp = Y(:,:,subj);
    Yvec(:,subj) = Ytemp(:);
end

tMat = zeros(niter,nsubj,size(t0,2));
tMat_exp = tMat;
invtMat = zeros(niter,nsubj,size(t0,2));
invtMat_exp = invtMat;

bMat = zeros(niter,nsubj);
XMat = zeros(niter, ncoord_l);
sigmaMat = zeros(niter, nsubj);
alphaMat = zeros(niter, 1);
rhoMat = zeros(niter, 1);
acceptMat = zeros(niter, nsubj, npar);
invacceptMat = zeros(niter, nsubj, npar);
acceptMatRho = zeros(niter, 1);
bi = ones(nsubj,1);

XTs = zeros(ncoord, nsubj);

s_beta = hyperPars.s_beta;
mu_beta= hyperPars.mu_beta;
sa=hyperPars.sa;
sb=hyperPars.sb;

ropt = hyperPars.ropt;
% rhat = hyperPars.rhat;
adapt = hyperPars.adapt;
k=hyperPars.k;
c0=hyperPars.c0;
c1=hyperPars.c1;
logsig = hyperPars.logsig;
Sig = hyperPars.Sig;
invlogsig = hyperPars.invlogsig;
invSig = hyperPars.invSig;

logsigRho = hyperPars.logsigRho;
bnds_rho = hyperPars.bnds_rho;
Xlatent_cut = cutTemplate(Xlatent, coord, coord_tmp);

DT_exp = repmat(eye(3), [1 1 nsubj]);
inv_DT_exp = repmat(eye(3), [1 1 nsubj]);

for iter = 1:niter
%     [~, DT_exp, ~, inv_DT_exp] = sym_updatesDerivative(Y, Xlatent_cut, b, t0, invt0, exp(hyperPars.logsig)/2, iter, 1e3, 1);

    % update transformation
%     tupdate1 = sym_transfUpdate_2d_parfor_mex(t0,invt0, DT_exp,inv_DT_exp, nnIDXs, Xlatent, Xlatent_cut, Y, sigma,...
%         alpha, rho, coord, coord_tmp, K, b, nnDict_L, logsig, Sig, invlogsig, invSig, Cov_NN, incre, nsubj, ...
%         hyperPars.ta0, hyperPars.tb0, hyperPars.priors,hyperPars.invpriors, hyperPars.l0, parworkers, hyperPars.lambda_sym);
%     tupdate1 = sym_transfUpdate_2d_parfor_mex(t0,invt0, DT_exp,inv_DT_exp, nnIDXs, Xlatent, Xlatent_cut, Y, sigma,...
%         alpha, rho, coord, coord_tmp, K, b, nnDict_L, logsig, Sig, invlogsig, invSig, Cov_NN, incre, nsubj, ...
%         hyperPars.ta0, hyperPars.tb0, hyperPars.priors,hyperPars.invpriors, hyperPars.l0, parworkers, hyperPars.lambda_sym, hyperPars.jointUpdates);
    tupdate1 = sym_transfUpdate_2d_parfor(t0,invt0, DT_exp,inv_DT_exp, nnIDXs, Xlatent, Xlatent_cut, Y, sigma,...
        alpha, rho, coord, coord_tmp, K, b, nnDict_L, logsig, Sig, invlogsig, invSig, Cov_NN, incre, nsubj, ...
        hyperPars.ta0, hyperPars.tb0, hyperPars.priors,hyperPars.invpriors, hyperPars.l0, parworkers, hyperPars.lambda_sym, hyperPars.jointUpdates);
    tupdate = sym_CenterTransforms_largeX(tupdate1, Y, nnIDXs, nsubj, coord, coord_tmp, incre, K, nnDict_L);

    nnIDXs = tupdate.nnIDXs;
    Y_inv = tupdate.Y_inv;
    t0 = tupdate.transf_exp;
    tMat(iter,:,:) = tupdate.transf_log;
    tMat_exp(iter,:,:) = tupdate.transf_exp;
    invtMat(iter,:,:) = tupdate.invtransf_log;
    invtMat_exp(iter,:,:) = tupdate.invtransf_exp;
    invt0 = tupdate.invtransf_exp;
    
    acceptMat(iter,:, :) = tupdate1.accept;
    invacceptMat(iter,:, :) = tupdate1.accept_inv;

    Y_inv_expand = expandMaps(Y_inv, L, nsubj);

    if iter > floor(niter/5)
        ropt = 0.234;
    end

    % adaptive MCMC
%     [logsig, Sig] = adaptiveMCMC(adapt, iter, k, tstart, tMat, acceptMat, ...
%         logsig, Sig, c0, c1, ropt, nsubj, npar);
%     [invlogsig, invSig] = adaptiveMCMC(adapt, iter, k, tstart, invtMat, invacceptMat, ...
%         invlogsig, invSig, c0, c1, ropt, nsubj, npar);
    % update X(T) and bi, sigma
        Xlatent_cut = cutTemplate(Xlatent, coord, coord_tmp);

    [Cov_NN, XTs, bi, sigma] = sym_update_XT_bi_sigma_mex(Y_inv, Xlatent, Xlatent_cut, ...
        ncoord, coord_tmp, nnIDXs, rho, K, nsubj, Cov_NN, ...
        Y, alpha, sigma, b, mu_beta, s_beta, sa, sb, parworkers);
    b = exp(log(bi)-mean(log(bi)))';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;

    % update alpha
    muFsX = CondLatentX_2d_mex(Xlatent, coord_tmp, rho, nnX_L, false, 1e-10);
    alpha = alphaUpdate_parallel_mex(Xlatent, XTs, nsubj, Cov_NN, muFsX, 1e-9, 1e-9, parworkers);
    alphaMat(iter) = alpha;

    % update latent map X
    Xlatent = sym_Xupdate_2d_parallel_mex(Y_inv_expand, b, sigma, XTs, Xlatent,...
        nnIDXs, nnX_L, nsubj, alpha, Cov_NN, coord_tmp, rho, parworkers);
    XMat(iter,:) = Xlatent';

    % update rho
    rho = RhoUpdate_2d_parallel_mex_v1(Xlatent, Yvec, b, sigma, rho, coord_tmp, ...
        nnIDXs, nsubj, K, alpha, bnds_rho,...
        exp(logsigRho), Cov_NN, muFsX, nnX_L, parworkers);
    rhoMat(iter) = rho.rho;
    Cov_NN = rho.Covs_nn_t;
    acceptMatRho(iter) = rho.accept;
    rho = rho.rho;
    logsigRho = adaptiveMCMC_rho(adapt, iter, k, acceptMatRho, c0, c1, ropt, logsigRho);

%     if rem(iter,5*k)==0
%         disp(iter);
%         disp(toc(tstart));
%         showOnefig(Xlatent, 1,iter)
%         drawnow
%         showSummaryFigs(XMat, 2, iter, 50)
%         drawnow
%         showOnefig(Xlatent_cut, 3, iter)
%         drawnow
%         showTransf(tMat_exp, 4, iter)
%         drawnow
%         showTransfinv(invtMat_exp, 5, iter)
%         drawnow
%         showYinv(Y_inv, 6)
%         drawnow
%         showsigmas(bMat, sigmaMat, 7, iter)
%         drawnow
% %         figure; showfig(XTs)
%     end

    if rem(iter,5*k)==0
        disp(iter);
        disp(toc(tstart));
        showOnefig(Xlatent, 1,iter)
        drawnow
        showSummaryFigs(XMat, 2, iter, 50)
        drawnow
        showOnefig(Xlatent_cut, 3, iter)
        drawnow
        showInverseImags(Y_inv, 4)
        drawnow
        showShifts(tMat_exp, 5, iter)
        drawnow
        showShifts(invtMat_exp, 6, iter)
        drawnow
    end

end

out.invtMat = invtMat;
out.invtMat_exp = invtMat_exp;
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



