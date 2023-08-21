function out = MCMC1d_rbf(init, dataIn, niter, hyperPars)

tstart = tic;
nsubj = dataIn.nsubj;
ncoord = length(dataIn.coord);
coord = dataIn.coord;
incre = coord(2) - coord(1);

[KY, KC] = rbf_1d(coord, hyperPars.KN, hyperPars.shape);


Xlatent = init.X;
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
nnX = dataIn.nnX;
nnDict = dataIn.nnDict;

t0 = [init.scaling; init.shift]';
coordTransf = t0 * [coord, ones(ncoord,1)]';
WKY = [];

for subj = 1:nsubj
%     nnIDXs(subj) = nns(nnDict, coord, coordTransf(subj, :)', incre, K);
%     Cov_NN(subj) = Cov_NN_Transfer(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
    
    WKY(:,:,subj) = warpRBF_1d(coord, coordTransf(subj,:)', KY);

end

tMat = zeros(niter,nsubj,2);
tMatLog = zeros(niter,nsubj,2);
bMat = zeros(niter,nsubj);
XMat = zeros(niter, ncoord);
sigmaMat = zeros(niter, 3);
alphaMat = zeros(niter, 1);
rhoMat = zeros(niter, 1);
acceptMat = zeros(niter, nsubj, 2);
acceptMatRho = zeros(niter, 1);

parworkers = min(nsubj, 8);
XTs = zeros(ncoord, nsubj);

s_beta = 1000;
mu_beta= 1;
bi = zeros(nsubj,1);
sa=1e-9;
sb=1e-9;

ropt = 0.234;
rhat = zeros(nsubj, 2);
adapt = true;
k=50;
c0=1;
c1=0.8;
logsig = repmat(log(2.4^2/2), nsubj, 2);
Sig = .1*ones(2,2,nsubj);
SigRho = 1e-5;
logsigRho = log(2.4^2)-4;
bnds_rho = [0, 10];
lambda_rho = 1e-2;

w = pinv(KY)*Xlatent;

for iter = 1:niter
    tupdate = transfUpdate_rbf_parallel(t0, Y, sigma, WKY,KY, w, coord, b, logsig, Sig,...
         nsubj, parworkers);
    t0 = tupdate.transf;
    tMat(iter,:,:) = tupdate.transf;
    tMatLog(iter,:,:) =  tupdate.transf; tMatLog(iter,:,1) = log(tupdate.transf(:,1));
    acceptMat(iter,:,:) = tupdate.accept;
%     nnIDXs = tupdate.nnIDXs;
    WKY = tupdate.WKY;
    % adaptive MCMC
    if (adapt && rem(iter, k)==0)
        disp(iter);
        disp(toc(tstart));
        
        for subj = 1:nsubj
            Sig0tHat = cov(squeeze(tMatLog((iter-k+1):iter, subj,:)));
            for j = 1:2
                rhat(subj, j) = mean(acceptMat((iter-k+1):iter, subj, j));
                gamma1 = 1/((floor(iter/k)+1)^c1);
                gamma2 = c0*gamma1;
                
                logsig(subj, j) = logsig(subj, j) + gamma2 *(rhat(subj, j)-ropt);
            end
            Sig(:,:,subj) = Sig(:,:,subj)+gamma1*(Sig0tHat-Sig(:,:,subj));
        end
    end
    
   [XTs, bi, sigma] = update_XT_bi_sigma_rbf_mex(WKY, w, ncoord,...
        nsubj, Y, sigma,mu_beta, s_beta, sa, sb, parworkers);
        
    b = (bi-mean(bi)+1)';
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    w = wUpdates(Y, WKY, KC, b, parworkers, sigma);
    Xlatent = KY*w;
    XMat(iter+1,:) = Xlatent';
end

out.Transf = tMat;
out.b = bMat;
out.X = XMat;
out.sigma = sigmaMat;
out.alpha = alphaMat;
out.rho = rhoMat;

out.time = toc(tstart);
end

