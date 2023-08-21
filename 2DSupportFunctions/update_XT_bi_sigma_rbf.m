function [ XTs, bi, sigma] = update_XT_bi_sigma_rbf(coord, coord_tmp,...
    WKY,w, ncoord, nsubj,Y, sigma,mu_beta,s_beta,sa, sb, parworkers)

XTs = zeros(size(Y, 1), nsubj);
bi = zeros(nsubj,1);

parfor(subj = 1:nsubj, parworkers)
%     Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
%     XTs(:, subj) = XTupdate(Y(:,subj), alpha, sigma(subj), b(subj), Cov_NN(subj));
    XTs(:, subj) = cutTemplate(WKY(:,:,subj) * w,coord, coord_tmp);
    s2bi = sum(XTs(:,subj).^2) + s_beta;
    mubi = 1/s2bi * (s_beta * mu_beta + dot(XTs(:,subj), Y(:,subj)));
    bi(subj) = abs(rand(1) * sqrt(sigma(subj)/s2bi) + mubi);
    sigma(subj) = 1/gamrnd(sa + ncoord/2, 1./(sb+0.5 * (sum(Y(:,subj).^2)+mu_beta^2*s_beta-mubi^2*s2bi)));
end

end

