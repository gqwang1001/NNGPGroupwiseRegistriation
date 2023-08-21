function out = condProbXTgivenXandTrans(X,Y, alpha, rho, coord, K, nnIDX)

Covs = Cov_NN_Transfer(X, coord, nnIDX, rho, K, 1e-10);
out = logNormalPdf(Y, Covs.mu, sqrt(Covs.Ft * alpha));

end

