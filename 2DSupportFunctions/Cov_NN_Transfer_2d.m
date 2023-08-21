function out = Cov_NN_Transfer_2d(Xlatent, coord, nnIDX, rho, K, eps)

% coord: coordinates of template images

    Cov_T_NT = exp(-rho * nnIDX.Dists);
    ncoord = size(nnIDX.Idxs, 1);
    Bt = zeros(ncoord, K);
    Ft = zeros(ncoord, 1);
    mu = zeros(ncoord, 1);
    
    for i = 1:ncoord
        Cov_nn_tf = exp(-rho * squareform(pdist(coord(nnIDX.Idxs(i,:), :))));
        Bt(i, :) = Cov_T_NT(i, :) / Cov_nn_tf;
        Ft(i) = 1 - dot(Bt(i,:), Cov_T_NT(i, :)) + eps;
        mu(i) = dot(Bt(i, :), Xlatent(nnIDX.Idxs(i, :))); 
    end
    
    out.Cov_T_NT = Cov_T_NT;
    out.Bt = Bt;
    out.Ft = Ft;
    out.mu = mu;

end

