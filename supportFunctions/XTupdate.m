function XT = XTupdate(Y, alpha, sigma, b, Cov_nn)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    
    mu = b*Y/sigma + Cov_nn.mu ./(alpha * Cov_nn.Ft);
    Vinv = b^2/sigma + 1./(alpha*Cov_nn.Ft);
    
    XT = diag(sqrt(1./Vinv)) * randn(length(Y), 1) + mu ./ Vinv; 
end