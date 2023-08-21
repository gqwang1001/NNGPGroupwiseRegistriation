function out = sym_logNormalPdf(Yinv, Latent, sigma_inv, Y, mu, Sigma)
    
    logliks_inv = -0.5*log(pi)-log(sigma_inv) - ((Yinv-Latent)./sigma_inv).^2 /2;
    logliks = -0.5*log(pi)-log(Sigma) - ((Y-mu)./Sigma).^2 /2;
    out = sum(logliks+logliks_inv);
    
end