function out = condProbYgivenXandTrans(X,Y, alpha, rho, coord, K, nnIDX, sigma, b)

Covs = Cov_NN_Transfer(X, coord, nnIDX, rho, K, 1e-10);
out = logNormalPdf(Y, b*Covs.mu, sqrt(b^2*Covs.Ft * alpha +sigma));
end