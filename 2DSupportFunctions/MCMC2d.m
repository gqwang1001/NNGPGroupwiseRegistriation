function out = MCMC2d(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = int32(dataIn.nsubj);
ncoord = length(dataIn.coord);
coord = dataIn.coord;
incre = coord(2) - coord(1);
parworkers = min(16, nsubj);

Xlatent = init.X;
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
nnX = dataIn.nnX;
nnDict = dataIn.nnDict;
bnds = hyperPars.bnds;
% t0 = [init.scaling; init.shift]';
% coordTransf = t0 * [coord, ones(ncoord,1)]';
t0 = [init.scaling, init.rot', init.shift];

for subj = 1:nsubj
    coordTransf = Transfer_Coord(init.scaling(subj, :), init.rot(subj), init.shift(subj,:), coord);
    nnIDXs(subj) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
    Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
end

tMat = zeros(niter,nsubj,size(t0,2));
tMat_adaptLog = zeros(niter,nsubj,size(t0,2));
bMat = zeros(niter,nsubj);
XMat = zeros(niter, ncoord);
sigmaMat = zeros(niter, nsubj);
alphaMat = zeros(niter, 1);
rhoMat = zeros(niter, 1);
acceptMat = zeros(niter, nsubj, size(t0,2));
acceptMatRho = zeros(niter, 1);
bi = zeros(nsubj,1);

XTs = zeros(ncoord, nsubj);

s_beta = hyperPars.s_beta;
mu_beta= hyperPars.mu_beta;
sa=hyperPars.sa;
sb=hyperPars.sb;
priors = hyperPars.priors;

ropt = hyperPars.ropt;
rhat = hyperPars.rhat;
adapt = hyperPars.adapt;
k=hyperPars.k;
c0=hyperPars.c0;
c1=hyperPars.c1;
logsig = hyperPars.logsig;
Sig = hyperPars.Sig;
logsigRho = hyperPars.logsigRho;
bnds_rho = hyperPars.bnds_rho;

for iter = 1:niter
    
    tupdate = transfUpdate_2d_parallel_mex(t0, nnIDXs, Xlatent, Y, sigma, alpha, rho, coord, K, b, nnDict, logsig, Sig, Cov_NN, incre, nsubj, bnds(1:3,:), priors, parworkers);
    
    t0 = tupdate.transf;
    tMat(iter,:,:) = tupdate.transf;
    acceptMat(iter,:,:) = tupdate.accept;
    tMat_adaptLog(iter,:,:) = tupdate.transf;
    tMat_adaptLog(iter,:, 1:2) = log(tupdate.transf(:, 1:2));
    tMat_adaptLog(iter,:, 3) = tupdate.transf(:, 3)./180*pi;
    
    nnIDXs = tupdate.nnIDXs;
    
    % adaptive MCMC
    if (adapt && rem(iter, k)==0)
        disp(iter);
        disp(toc(tstart));
        for subj = 1:nsubj
            Sig0tHat = cov(squeeze(tMat_adaptLog((iter-k+1):iter,subj,:)));
            for j = 1:size(t0,2)
                rhat(subj, j) = mean(acceptMat((iter-k+1):iter, subj, j));
                gamma1 = 1/((floor(iter/k)+1)^c1);
                gamma2 = c0*gamma1;
                
                logsig(subj, j) = logsig(subj, j) + gamma2 *(rhat(subj, j)-ropt);
            end
            Sig(:,:,subj) = Sig(:,:,subj)+gamma1*(Sig0tHat-Sig(:,:,subj));
        end
    end
    
    [Cov_NN, XTs, bi, sigma] = update_XT_bi_sigma_mex(Xlatent, ncoord, coord, nnIDXs, rho, K, nsubj, Cov_NN, ...
        Y, alpha, sigma, b,mu_beta, s_beta,sa, sb, parworkers);
    
    b = (bi-mean(bi)+1)';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    muFsX = CondLatentX_2d_mex(Xlatent, coord, rho, nnX);
    alpha = alphaUpdate_parallel_mex(Xlatent, XTs, nsubj, Cov_NN, muFsX, 1e-9, 1e-9, parworkers);
    
    alphaMat(iter) = alpha;
    
    Xlatent = Xupdate_2d_parallel_mex(XTs, Xlatent, nnIDXs, nnX, nsubj, alpha, Cov_NN, coord, rho, parworkers);
    XMat(iter,:) = Xlatent';
    
    rho = RhoUpdate_2d_parallel_mex(Xlatent, XTs, rho, coord, nnIDXs, nsubj, K, alpha, [0,10],...
        exp(logsigRho), Cov_NN, muFsX, nnX, parworkers);
    rhoMat(iter) = rho.rho;
    Cov_NN = rho.Covs_nn_t;  
    acceptMatRho(iter) = rho.accept;

    rho = rho.rho;
    if (adapt && rem(iter, k)==0)
        rhatRho = mean(acceptMatRho((iter-k+1):iter));
        gamma1 = 1/((floor(iter/k)+1)^c1);
        gamma2 = c0*gamma1;
        logsigRho = logsigRho+gamma2*(rhatRho-ropt);
    end
end


out.Transf = tMat;
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



