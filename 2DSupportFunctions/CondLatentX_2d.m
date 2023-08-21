function out = CondLatentX_2d(X, coord, rho, nnX, CholMat, tau)
    V = length(X);
    mu = zeros(V, 1);
    Fs = ones(V, 1);
    expDists = exp(-rho * nnX.Dists);
    B_s = zeros(size(expDists));
    
    % i=2
    B_s(2, 1) = expDists(2, 1)/(1+tau);
    Fs(2) = 1+tau-expDists(2, 1)^2;
    mu(2) = B_s(2,1) * X(nnX.Idxs(2,1));
    
    for i = 3:V
        idxs = nnX.Idxs(i, :);
        iidxs = nnX.Idxs(i, :)>0;
        idxs = idxs(iidxs);
        COV = exp(-rho*squareform(pdist(coord(idxs, :))))+tau*eye(length(idxs));
        B_s(i, iidxs) =  expDists(i, iidxs)/COV;
        Fs(i) = 1+tau - dot(B_s(i, iidxs), expDists(i,iidxs));
        mu(i) = dot(B_s(i, iidxs), X(idxs));
    end
    
    out.Bs = B_s;
    out.mu = mu;
    out.Fs = Fs;
    out.A = nan;
    out.Dinv = nan;
    if CholMat==true
        Dinv = eye(V);
        A = zeros(V);
        for i = 2:V
            idx = nnX.Idxs(i,:);
            iidx = idx>0;
            idx = idx(iidx);
            A(i,idx) = B_s(i, iidx);
            Dinv(i,i) = 1/Fs(i); 
        end
        out.A = A;
        out.Dinv = Dinv;
    end
    
end

