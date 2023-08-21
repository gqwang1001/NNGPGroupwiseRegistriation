function out = RhoUpdate_2d_parallel(X, Y,bi, sigma, rho0, coord, nnIDXs, nsubj, K, alpha, bnds, lambda, Covs_nn_t, muFs, nnX, parworkers)

rho1 = rho0 + lambda * (-1 + 2* rand(1));

p01 = 0; % pdf of rho0 given rho1
p10 = 0; % pdf of rho1 given rho0

if (rho0>bnds(1)) && (rho0<bnds(2))
    p01 = unifpdf(rho0, rho1-lambda, rho1+lambda);
end

if (rho1>bnds(1)) && (rho1<bnds(2))
    p10 = unifpdf(rho1, rho0-lambda, rho0+lambda);
end

out.rho = rho0;
out.Covs_nn_t = Covs_nn_t;
% out.muFs = muFs;
out.accept = 0;

if p10 * p01 ~=0
    
    logliks0 = nan(nsubj,1);
    logliks1 = nan(nsubj,1);
    
    Covs_nn_t0 = Covs_nn_t;
    Covs_nn_t1 = Covs_nn_t;
    
    parfor (subj = 1:nsubj, parworkers)
        logliks0(subj) = logNormalPdf(Y(:,subj), bi(subj)*Covs_nn_t0(subj).mu, sqrt(bi(subj)^2*Covs_nn_t0(subj).Ft * alpha+sigma(subj)));
        
        Covs_nn_t1(subj) = Cov_NN_Transfer_2d(X, coord, nnIDXs(subj), rho1, K, 1e-10);
        logliks1(subj) = logNormalPdf(Y(:,subj), bi(subj)*Covs_nn_t1(subj).mu, sqrt(bi(subj)^2*Covs_nn_t1(subj).Ft * alpha+sigma(subj)));
    end
    
    logliksSum0 = sum(logliks0);
    logliksSum1 = sum(logliks1);
    
    muFs0 = muFs;
    logliksX0 = logNormalPdf(X, muFs0.mu, sqrt(muFs0.Fs * alpha));
    
    muFs1 = CondLatentX_2d(X, coord, rho1, nnX, false, 1e-9);
    logliksX1 = logNormalPdf(X, muFs1.mu, sqrt(muFs1.Fs * alpha));
    
    A = exp(logliksSum1-logliksSum0+logliksX1-logliksX0) * p01/p10;
    
    if rand(1)<A
        out.rho = rho1;
        out.Covs_nn_t = Covs_nn_t1;
%         out.muFs = muFs1;
        out.accept = 1;
    end
end

end