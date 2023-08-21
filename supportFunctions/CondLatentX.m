function out = CondLatentX(X, coord, rho, nnX)
    V = length(X);
    mu = zeros(V, 1);
    Fs = ones(V, 1);
    expDists = exp(-rho * nnX.Dists);
    B_s = zeros(size(expDists));
    
    % i=2
    B_s(2, 1) = expDists(2, 1);
    Fs(2) = 1 - expDists(2, 1)^2;
    mu(2) = B_s(2,1) * X(nnX.Idxs(2,1));
    
    for i = 3:V
        idxs = nnX.Idxs(i, :);
        iidxs = nnX.Idxs(i, :)>0;
        idxs = idxs(iidxs);
        
        B_s(i, iidxs) =  expDists(i, iidxs) / exp(-rho * squareform(pdist(coord(idxs))));
        Fs(i) = 1 - dot(B_s(i, iidxs), expDists(i,iidxs));
        mu(i) = dot(B_s(i, iidxs), X(idxs));
    end
    
    out.Bs = B_s;
    out.mu = mu;
    out.Fs = Fs;

end

