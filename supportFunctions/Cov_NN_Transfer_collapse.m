function out = Cov_NN_Transfer_collapse(Xlatent,coord, nnIDX, rho, K, sigma, b, alpha)

    Cov_T_NT = alpha*exp(-rho * nnIDX.Dists);
    ncoord = length(coord);
    Bt = zeros(ncoord, K);
    Ft = zeros(ncoord, 1);
    mu = zeros(ncoord, 1);
    
    for i = 1:ncoord
        Cov_nn_tf = alpha*exp(-rho * squareform(pdist(coord(nnIDX.Idxs(i,:)))));
        Bt(i, :) = Cov_T_NT(i, :)/Cov_nn_tf;
        Ft(i) = alpha*b^2-b^2*dot(Bt(i,:), Cov_T_NT(i, :))+sigma;
        mu(i) = b*dot(Bt(i,:),Xlatent(nnIDX.Idxs(i,:))); 
    end
    
    out.Cov_T_NT = Cov_T_NT;
    out.Bt = Bt;
    out.Ft = Ft/alpha;
    out.mu = mu;
end

