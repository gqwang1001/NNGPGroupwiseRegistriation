function out = MCMC1d(init, dataIn, niter)

tstart = tic;
nsubj = dataIn.nsubj;
ncoord = length(dataIn.coord);
coord = dataIn.coord;
incre = coord(2) - coord(1);

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

for subj = 1:nsubj
    nnIDXs(subj) = nns(nnDict, coord, coordTransf(subj, :)', incre, K);
    Cov_NN(subj) = Cov_NN_Transfer(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
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

s_beta = 1;
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

for iter = 1:niter
    tupdate = transfUpdate_parallel_mex(t0, nnIDXs, Xlatent, Y, sigma, alpha, rho, coord, K, b, nnDict, logsig, Sig, Cov_NN, incre, nsubj, parworkers);
    t0 = tupdate.transf;
    tMat(iter,:,:) = tupdate.transf;
    tMatLog(iter,:,:) =  tupdate.transf; tMatLog(iter,:,1) = log(tupdate.transf(:,1));
    acceptMat(iter,:,:) = tupdate.accept;
    nnIDXs = tupdate.nnIDXs;
    
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
    
    for subj = 1:nsubj
        Cov_NN(subj) = Cov_NN_Transfer_mex(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
        XTs(:, subj) = XTupdate_mex(Y(:,subj), alpha, sigma(subj), b(subj), Cov_NN(subj));
        
        s2bi = sum(XTs(:,subj).^2) + s_beta;
        mubi = 1/s2bi * (s_beta * mu_beta + dot(XTs(:,subj), Y(:,subj)));
        bi(subj) = rand(1) * sqrt(sigma(subj)/s2bi) + mubi;
        sigma(subj) = 1/gamrnd(sa + ncoord/2, 1./(sb+0.5 * (sum(Y(:,subj).^2)+mu_beta^2*s_beta-mubi^2*s2bi)));
    end
    
    b = bi-mean(bi)+1;
    bMat(iter,:) = b;
    sigmaMat(iter,:) = sigma;
    
    muFsX = CondLatentX(Xlatent, coord, rho, nnX);
    alpha = alphaUpdate_parallel_mex(Xlatent, XTs, nsubj, Cov_NN, muFsX, 1e-9, 1e-9, parworkers);
    alphaMat(iter) = alpha;
    
    Xlatent = Xupdate_collapse_mex(Y, Xlatent, nnIDXs, nnX, nsubj, alpha, sigma, b, Cov_NN, coord, rho);
    XMat(iter,:) = Xlatent';
    
    rhostr = RhoUpdate_mex(Xlatent, Y, rho, coord, nnIDXs, nsubj, K, alpha, bnds_rho, ...
        exp(logsigRho),b,Cov_NN, muFsX, nnX, sigma);
    %     rhostr = RhoUpdate_adaptive_mex(Xlatent, Y, rho, coord, nnIDXs, nsubj, K, alpha,b, 10,...
    %         logsigRho, SigRho, Cov_NN, muFsX, nnX, sigma);
    
    rhoMat(iter) = rhostr.rho;
    Cov_NN = rhostr.Covs_nn_t;
    rho = rhostr.rho;
    acceptMatRho(iter) = rhostr.accept;

    if (adapt && rem(iter, k)==0)
        
%         SigRho0tHat = var(log(rhoMat((iter-k+1):iter)));
        rhatRho = mean(acceptMatRho((iter-k+1):iter));
        gamma1 = 1/((floor(iter/k)+1)^c1);
        gamma2 = c0*gamma1;
        
        logsigRho = logsigRho+gamma2*(rhatRho-ropt);
%         SigRho = SigRho+gamma1*(SigRho0tHat-SigRho);
        %         disp([logsigRho, SigRho*1e5]);
    end
end

out.Transf = tMat;
out.b = bMat;
out.X = XMat;
out.sigma = sigmaMat;
out.alpha = alphaMat;
out.rho = rhoMat;

out.time = toc(tstart);
end

