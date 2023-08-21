function out = MCMC2d_grouping(init, dataIn, niter)

tstart = tic;
nsubj = dataIn.nsubj;
ncoord = length(dataIn.coord);
coord = dataIn.coord;
incre = coord(2) - coord(1);
parworkers = min(4, nsubj);

Xlatent = init.X;
K = dataIn.K;
rho = init.rho;
alpha = init.alpha;
Y = dataIn.Y;
b = init.b;
sigma = init.sigma;
nnX = dataIn.nnX;
nnDict = dataIn.nnDict;
bnds = dataIn.bnds;
pz = init.p;
nclass = dataIn.nclass;

% t0 = [init.scaling; init.shift]';
% coordTransf = t0 * [coord, ones(ncoord,1)]';
t0 = [];
for subj = 1:nsubj
    for c = 1:nclass
        t0(c,:,:) = [squeeze(init.scaling(c,:,:))', init.rot(c,:)', squeeze(init.shift(c,:,:))'];
        coordTransf = Transfer_Coord(init.scaling(c,:,subj), init.rot(c,subj), init.shift(c,:,subj), coord);
        nnIDXs(subj,c) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
        Cov_NN(subj,c) = Cov_NN_Transfer_2d(Xlatent(:,c), coord, nnIDXs(subj,c), rho(c), K, 1e-10);
    end
end

tMat = zeros(nclass,niter,nsubj,size(t0,2));
tMat_adaptLog = zeros(nclass,niter,nsubj,size(t0,2));
bMat = zeros(nclass,niter,nsubj);
XMat = zeros(nclass,niter, ncoord);
sigmaMat = zeros(nclass,niter, nsubj);
alphaMat = zeros(nclass,niter, 1);
rhoMat = zeros(nclass,niter, 1);
acceptMat = zeros(nclass,niter, nsubj, size(t0,2));

XTs = zeros(nclass, ncoord, nsubj);

s_beta = 1;
mu_beta= 1;
bi = zeros(nclass,nsubj,1);
sa=1e-9;
sb=1e-9;

ropt = 0.234;
rhat = zeros(nclass,nsubj,size(t0,2));
adapt = true;
k=100;
c0=1;
c1=0.8;
logsig = repmat(log(2.4^2/2), nclass, nsubj, 5);
Sig = 1*ones(nclass, 5,5,nsubj);
Sig(:, 1:3, 1:3, :) = .1*ones(nclass,3,3,nsubj);

for iter = 1:niter
    
    for c=1:nclass
    
    tupdate = transfUpdate_2d_parallel(t0(c,:,:), nnIDXs(:,c), Xlatent(:,c), Y, sigma(c,:),...
              alpha(c), rho(c), coord, K, b(c,:), nnDict, logsig(c,:,:), Sig(c,:,:,:),Cov_NN(:,c), ...
              incre, nsubj, bnds, parworkers);
    
    t0(c,:,:) = tupdate.transf;
    nnIDXs(:,c) = tupdate.nnIDXs;
    tMat(c,iter,:,:) = tupdate.transf;
    
    acceptMat(c, iter,:,:) = tupdate.accept;
    tMat_adaptLog(c,iter,:,:) = tupdate.transf;
    tMat_adaptLog(c,iter,:, 1:2) = log(tupdate.transf(:,1:2));
    tMat_adaptLog(c,iter,:, 3) = tupdate.transf(:,3)./180*pi;
    
    
    % adaptive MCMC
    if (adapt && rem(iter, k)==0)
        disp(iter);
        disp(toc(tstart));
        for subj = 1:nsubj
            Sig0tHat = cov(squeeze(tMat_adaptLog(c,(iter-k+1):iter,subj,:)));
            for j = 1:size(t0,2)
                rhat(subj, j) = mean(squeeze(acceptMat(c, (iter-k+1):iter, subj, j)));
                gamma1 = 1/((floor(iter/k)+1)^c1);
                gamma2 = c0*gamma1;
                
                logsig(c,subj, j) = logsig(c,subj, j) + gamma2 *(rhat(c,subj, j)-ropt);
            end
            Sig(c,:,:,subj) = Sig(c,:,:,subj)+gamma1*(Sig0tHat-Sig(c,:,:,subj));
        end
    end
    [Cov_NN_c, XTs_c, bi_c, sigma_c] = update_XT_bi_sigma(Xlatent(:,c), ncoord, coord, nnIDXs(:,c), rho(c), K, nsubj, Cov_NN(:,c), ...
                                Y, alpha(c), sigma(c,:), b(c,:), mu_beta, s_beta,sa, sb, parworkers);
    Cov_NN(:,c) = Cov_NN_c;
    XTs(c,:,:) = XTs_c;
    bi(c,:) = bi_c;
    sigma(c,:) = sigma_c;
                            
    b(c,:) = (bi(c,:)-mean(bi(c,:))+1)';
    bMat(c,iter,:) = b(c,:);
    sigmaMat(c,iter,:) = sigma(c,:);
    
    muFsX = CondLatentX_2d(Xlatent(:,c), coord, rho(c), nnX);
    alpha(c) = alphaUpdate_parallel_mex(Xlatent(:,c), XTs_c, nsubj, Cov_NN_c, muFsX, 1e-9, 1e-9, parworkers);
    
    alphaMat(c,iter) = alpha(c);
    
    Xlatent_c = Xupdate_2d_parallel_mex(XTs_c, Xlatent(:,c), nnIDXs(:,c), nnX, nsubj, alpha(c), Cov_NN_c, coord, rho(c), parworkers);
    XMat(c,iter,:) = Xlatent_c';
    
    rho = RhoUpdate_2d_parallel_mex(Xlatent_c, XTs_c, rho(c), coord, nnIDXs(:,c), nsubj, K, alpha(c), [0,2],...
                                    5e-3, Cov_NN_c, muFsX, nnX, parworkers);
    rhoMat(c,iter) = rho.rho;
    Cov_NN(:,c) = rho.Covs_nn_t;
    rho(c) = rho.rho;
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



