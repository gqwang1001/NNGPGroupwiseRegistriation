function y = my_mvtpdf(X, C, df)

[n,d] = size(X);

% Standardize C to correlation if necessary.  This does NOT standardize X.
s = sqrt(diag(C));
if (any(s~=1))
    C = C ./ (s * s');
end

% Make sure C is a valid covariance matrix
R = chol(C);

df = df(:);

% Create array of standardized data, and compute log(sqrt(det(Sigma)))
Z = X / R;
logSqrtDetC = sum(log(diag(R)));

logNumer = -((df+d)/2) .* log(1+sum(Z.^2, 2)./df);
logDenom = logSqrtDetC + (d/2)*log(df*pi);
y = exp(gammaln((df+d)/2) - gammaln(df/2) + logNumer - logDenom);
