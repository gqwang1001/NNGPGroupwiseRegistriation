function out = my_mvnrnd(mu, Sig)
n = length(mu);
R = chol(Sig);
out = mu + randn(1, n) * R;
end