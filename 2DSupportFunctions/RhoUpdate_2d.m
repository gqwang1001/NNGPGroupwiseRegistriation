function out = RhoUpdate_2d(X, Y, rho0, coord, nnIDXs, nsubj, K, alpha, bnds, lambda, Covs_nn_t, muFs, nnX)
    
    rho1 = rho0 + lambda * (-1 + 2* rand(1));
    
    p01 = 0; % pdf of rho0 given rho1
    p10 = 0; % pdf of rho1 given rho0
    
    if (rho0>bnds(1)) && (rho0<bnds(2))
        p01 = unifpdf(rho0, rho1-lambda, rho1+lambda);
    end
    
    if (rho1>bnds(1)) && (rho1<bnds(2))
        p10 = unifpdf(rho1, rho0-lambda, rho0+lambda);
    end
    
    
    if p10 * p01 ==0
        out.rho = rho0;
        out.Covs_nn_t = Covs_nn_t;
        out.muFs = muFs;
    else
        logliks0 = nan(nsubj,1);
        logliks1 = nan(nsubj,1);
        
        Covs_nn_t0 = Covs_nn_t;
        Covs_nn_t1 = Covs_nn_t;
        
        for subj = 1:nsubj
            logliks0(subj) = logNormalPdf(Y(:,subj), Covs_nn_t0(subj).mu, sqrt(Covs_nn_t0(subj).Ft * alpha));
        
            Covs_nn_t1(subj) = Cov_NN_Transfer_2d(X, coord, nnIDXs(subj), rho1, K, 1e-10);
            logliks1(subj) = logNormalPdf(Y(:,subj), Covs_nn_t1(subj).mu, sqrt(Covs_nn_t1(subj).Ft * alpha));
        end
        
        logliksSum0 = sum(logliks0);
        logliksSum1 = sum(logliks1);
        
        muFs0 = muFs;
        logliksX0 = logNormalPdf(X, muFs0.mu, sqrt(muFs0.Fs * alpha));
        
        muFs1 = CondLatentX_2d(X, coord, rho1, nnX);
        logliksX1 = logNormalPdf(X, muFs1.mu, sqrt(muFs1.Fs * alpha));
        
        A = exp(logliksSum1-logliksSum0+logliksX1-logliksX0) * p01/p10;
        
        out.rho = rho0;
        out.Covs_nn_t = Covs_nn_t0;
        out.muFs = muFs0;
         
        if rand(1)<A
             out.rho = rho1;
             out.Covs_nn_t = Covs_nn_t1;
             out.muFs = muFs1;
        end
    end
        
        
end