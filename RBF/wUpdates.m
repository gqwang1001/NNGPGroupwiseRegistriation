function out = wUpdates(Y, WKY, KC, b, parworkers, sigma)

nc = size(KC, 1);
nsubj = size(WKY, 3);

muw1 = zeros(nc, nsubj);
Sw1 = zeros(nc, nc, nsubj);

% parfor(i=1:nsubj, parworkers)    
for i=1:nsubj
    Sw1(:,:,i) = WKY(:,:,i)'* WKY(:,:,i) * b(i)^2/sigma(i);
    muw1(:,i) = b(i)*WKY(:,:,i)'*Y(:,i)/sigma(i);
end

Sw = KC + sum(Sw1, 3)+1e-8*eye(size(Sw1,1));
invSw = inv(Sw)+1e-8*eye(size(Sw,1));
muw = invSw*sum(muw1,2);

out = my_mvnrnd(muw', invSw);
out = out';

end

function out = my_mvnrnd(mu, Sig)
n = length(mu);
R = chol(Sig);
out = mu + randn(1, n) * R;
end