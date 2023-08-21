function out = loglikLatentMap(X, alpha, rho, coord, nnX)

    muFs = CondLatentX(X, coord, rho, nnX);
    out = logNormalPdf(X, muFs.mu, sqrt(muFs.Ft * alpha));

end

