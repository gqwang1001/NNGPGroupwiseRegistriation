function out = logNormalPdf(X, mu, Sigma)
    Xstd = (X-mu)./Sigma;
    logliks = -0.5 * log(2*pi)-log(Sigma) - Xstd.^2 /2;
    out = sum(logliks);
end