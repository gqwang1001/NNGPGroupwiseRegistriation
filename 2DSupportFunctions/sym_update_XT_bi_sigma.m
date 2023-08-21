function [Cov_NN, XTs, bi, sigma] = sym_update_XT_bi_sigma(YT_inv, Xlatent, X_cut,ncoord, coord, nnIDXs, rho, K, nsubj,Cov_NN,...
    Ymat, alpha, sigma, b, mu_beta,...
    s_beta,sa, sb, parworkers)

S = size(Ymat);
% Y = reshape(Ymat,[S(1)*S(2),S(3)]);
XTs = zeros(S(1)*S(2), nsubj);
bi = zeros(nsubj,1);

sumXLatentSquared = sum(X_cut.^2);

parfor(subj = 1:nsubj, parworkers)
    Yvec1 = Ymat(:,:,subj); Yvec = Yvec1(:);
    YInv_vec1 = YT_inv(:,:,subj); YInv_vec = YInv_vec1(:);
    
    Cov_NN(subj) = Cov_NN_Transfer_2d(Xlatent, coord, nnIDXs(subj), rho, K, 1e-10);
    XTs(:, subj) = XTupdate(Yvec, alpha, sigma(subj), b(subj), Cov_NN(subj));
    
    s2bi = sumXLatentSquared+sum(XTs(:,subj).^2) + s_beta;
    mubi = 1/s2bi * (s_beta * mu_beta + dot(XTs(:,subj), Yvec)+dot(X_cut, YInv_vec));
    bi(subj) = rand(1) * sqrt(sigma(subj)/s2bi) + mubi;
    sigma(subj) = 1/gamrnd(sa + ncoord,...
        1./(sb+0.5 * (sum(Yvec.^2)+sum(YInv_vec.^2)+mu_beta^2*s_beta-mubi^2*s2bi)));
    bi(subj) = abs(bi(subj));
end

end

