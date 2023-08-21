function out = RhoUpdate(X, Y, rho0, coord, nnIDXs, nsubj, K, alpha, bnds, ...
    lambda,b,Covs_nn_t, muFs, nnX, sigma)

rho1 = rho0+lambda*(-1+2*rand(1));

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
out.muFs = muFs;
out.accept = 0;

if p10*p01~=0
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
    
    logliksX0 = logNormalPdf(X, muFs.mu, sqrt(muFs.Fs*alpha));
    
    muFs1 = CondLatentX(X, coord, rho1, nnX);
    logliksX1 = logNormalPdf(X, muFs1.mu, sqrt(muFs1.Fs*alpha));
    
    A = exp(sum(logliks1)-sum(logliks0)+logliksX1-logliksX0)*p01/p10;
    
    if rand(1)<A
        out.rho = rho1;
        out.Covs_nn_t = Covs_nn_t1;
        out.muFs = muFs1;
        out.accept = 1;
    end
end

end