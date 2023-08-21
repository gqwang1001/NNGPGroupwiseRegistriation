function out = RhoUpdate_adaptive(X, Y, rho0, coord, nnIDXs, nsubj, K, ...
    alpha,b, bnds, logsig, Sig, Covs_nn_t, muFs, nnX, sigma)

logrho1 = randn(1)*sqrt(exp(logsig)*Sig)+log(rho0);
rho1 = exp(logrho1);

out.rho = rho0;
out.Covs_nn_t = Covs_nn_t;
out.muFs = muFs;
out.accept = 0;

if rho1<bnds
    
    logliks0 = zeros(nsubj,1);
    logliks1 = zeros(nsubj,1);
    
    Covs_nn_t1 = Covs_nn_t;
    
    for subj = 1:nsubj
        logliks0(subj) = logNormalPdf(Y(:,subj), b(subj)*Covs_nn_t(subj).mu,...
            sqrt(b(subj)^2*Covs_nn_t(subj).Ft*alpha+sigma(subj)));
        
        Covs_nn_t1(subj) = Cov_NN_Transfer(X, coord, nnIDXs(subj), rho1, K, 1e-10);
        logliks1(subj) = logNormalPdf(Y(:,subj), b(subj)*Covs_nn_t1(subj).mu, ...
            sqrt(b(subj)^2*Covs_nn_t1(subj).Ft*alpha+sigma(subj)));
    end
    
    logliksSum0 = sum(logliks0);
    logliksSum1 = sum(logliks1);
    
    logliksX0 = logNormalPdf(X, muFs.mu, sqrt(muFs.Fs*alpha));
    
    muFs1 = CondLatentX(X, coord, rho1, nnX);
    logliksX1 = logNormalPdf(X, muFs1.mu, sqrt(muFs1.Fs*alpha));
    
    A = exp(logliksSum1-logliksSum0+logliksX1-logliksX0);
    
    if rand(1)<A
        out.rho = rho1;
        out.Covs_nn_t = Covs_nn_t1;
        out.muFs = muFs1;
        out.accept = 1;
    end
end

end