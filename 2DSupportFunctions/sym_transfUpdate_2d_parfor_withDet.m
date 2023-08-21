function out = sym_transfUpdate_2d_parfor_withDet(transf0, invtransf0,  DTs, invDTs, nnIDXs, X, X_cut, Y, sigma,...
    alpha, rho, coord,coord_tmp, K, b, nnDict, logsig, Sig, invlogsig, invSig, Cov_NN0, incre, nsubj, ...
    ta0, tb0, priors, invpriors, lambda, parworkers, lambda_sym, jointUpdate)

logliks0 = zeros(nsubj,1);
logliks1 = zeros(nsubj,1);

invlogliks0 = zeros(nsubj,1);
invlogliks1 = zeros(nsubj,1);

% Y_inv = zeros(size(Y,1), size(Y,2), nsubj);
accept = zeros(nsubj, 6);
accept_inv = zeros(nsubj, 6);

t1 = transf0; % Tmat
inv_t1 = invtransf0; % Tmat

nnIDXnew = nnIDXs;
Covs_NN1 = Cov_NN0;

parfor(i=1:nsubj, parworkers)
% for i =1:nsubj
    %update t_inv
    t1Mat0 = Vec2Mat(t1(i,:), false);
    invt1Mat0 = Vec2Mat(inv_t1(i,:), false);
    invTvec = invt1Mat0(:);
    Y_i = Y(:,:,i);
    accept_inv_i = zeros(1,6);
    accept_i=zeros(1, 6);
    
    if jointUpdate == true % jointUpdate
        
        invprop0 = my_mvnrnd(zeros(1, 6), exp(invlogsig(i,1))*invSig(1:6,1:6,i));
        invTmat_prop = expm(Vec2Mat(invprop0, true))*invDTs(:,:,i)*invt1Mat0;
        
        if ~checkLogMat(invTmat_prop) && ~checkLogMat(invTmat_prop/(Vec2Mat(invpriors(i,:), false)))
            
            Yinv_i = Warp_affine_mat(invt1Mat0, Y_i);
            invlogliks0(i) = logNormalPdf(Yinv_i(:), b(i)*X_cut, sqrt(sigma(i)))+...
                logMVTdistPrior(invTvec, invpriors(i,:), ta0, tb0, lambda)-...
                (lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'));
            
            invTvec_prop = invTmat_prop(:);
            Yinv_i1 = Warp_affine_mat(invTmat_prop, Y_i);
            invlogliks1(i) = logNormalPdf(Yinv_i1(:), b(i)*X_cut, sqrt(sigma(i)))+...
                logMVTdistPrior(invTvec_prop, invpriors(i,:), ta0, tb0, lambda)-...
                (lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'));
%             AP = min(0, invlogliks1(i)-invlogliks0(i)+ logProposalDiff(invTmat_prop, invt1Mat0, invSig(1:6,1:6,i)));
            AP = min(0, invlogliks1(i)-invlogliks0(i));

            if (log(rand(1)) < AP)
                inv_t1(i,:) = invTvec_prop(1:6)';
                accept_inv_i(1) = 1;
            end
        end
        
        % update t
        prop0 = my_mvnrnd(zeros(1, 6), exp(logsig(i,1))*Sig(1:6,1:6,i));
        Tmat_prop = expm(Vec2Mat(prop0, true))*DTs(:,:,i)*t1Mat0;
        if ~checkLogMat(Tmat_prop) && ~checkLogMat(Tmat_prop/(Vec2Mat(priors(i,:), false)))
            Tvec = t1Mat0(:);
            invt1Mat0 = Vec2Mat(inv_t1(i,:), false);
            logliks0(i) = logNormalPdf(Y_i(:), b(i)*Cov_NN0(i).mu, sqrt(b(i)^2*Cov_NN0(i).Ft * alpha + sigma(i)))+...
                logMVTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda)-...
                (lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'));
            
            Tvecprop = Tmat_prop(:);
            coordTransf = Transform_mat_coord_affine(Tmat_prop, coord);
            nnIDXnew(i) = nns_2d_square(nnDict, coord_tmp, coordTransf, incre, K);
            Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord_tmp, nnIDXnew(i), rho, K, 1e-10);
            logliks1(i) = logNormalPdf(Y_i(:), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha + sigma(i)))+...
                logMVTdistPrior(Tvecprop, priors(i,:), ta0, tb0, lambda)-...
                (lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'));
            
%             AP = min(0, logliks1(i)-logliks0(i)+ logProposalDiff(Tmat_prop, t1Mat0, Sig(1:6,1:6,i)));
            AP = min(0, logliks1(i)-logliks0(i));

            if (log(rand(1)) < AP)
                t1(i,:) = Tvecprop(1:6)';
                accept_i(1) = 1;
            end
        end
        
    else % jointUpdate  false
        
        Yinv_i = Warp_affine_mat(invt1Mat0, Y_i);
        invlogliks0(i) = logNormalPdf(Yinv_i(:), b(i)*X_cut, sqrt(sigma(i)))+...
            logMVTdistPrior(invTvec, invpriors(i,:), ta0, tb0, lambda)-...
            (lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'));
        invprop0 = zeros(1, 6);
        
        for j=1:6
            invprop0(j) = randn(1)*sqrt(exp(invlogsig(i,j))*invSig(j,j,i));
            invTmat_prop = expm(Vec2Mat(invprop0, true))*invDTs(:,:,i)*invt1Mat0;
            
            if ~checkLogMat(invTmat_prop) && ~checkLogMat(invTmat_prop/(Vec2Mat(invpriors(i,:), false)))
                invTvec_prop = invTmat_prop(:);
                Yinv_i1 = Warp_affine_mat(invTmat_prop, Y_i);
                invlogliks1(i) = logNormalPdf(Yinv_i1(:), b(i)*X_cut, sqrt(sigma(i)))...
                    -(lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'))...
                    +logMVTdistPrior(invTvec_prop, invpriors(i,:), ta0, tb0, lambda);
                
                AP = min(0, invlogliks1(i)-invlogliks0(i) + logProposalDiff(invTmat_prop, invt1Mat0, diag(diag(invSig(1:6,1:6,i)))));
%                 AP = min(0, invlogliks1(i)-invlogliks0(i));


                if log(rand(1)) < AP
                    inv_t1(i,:) = invTvec_prop(1:6)';
                    accept_inv_i(j) = 1;
                    invlogliks0(i)=invlogliks1(i);
                else
                    invprop0(j) = 0;
                end
            end
        end
        accept_inv(i,:) = accept_inv_i;
        % update t
        
        Tvec = t1Mat0(:);
        invt1Mat0 = Vec2Mat(inv_t1(i,:), false);
        logliks0(i) = logNormalPdf(Y_i(:), b(i)*Cov_NN0(i).mu, sqrt(b(i)^2*Cov_NN0(i).Ft * alpha + sigma(i)))+...
            logMVTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda)-...
            (lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'));
        prop0 = zeros(1, 6);
        for j = 1:6
            prop0(j) = randn(1)*sqrt(exp(logsig(i,j))*Sig(j,j,i));
            Tmat_prop = expm(Vec2Mat(prop0, true))*DTs(:,:,i)*t1Mat0;
            
            if ~checkLogMat(Tmat_prop) && ~checkLogMat(Tmat_prop/(Vec2Mat(priors(i,:), false)))
                Tvecprop = Tmat_prop(:);
                coordTransf = Transform_mat_coord_affine(Tmat_prop, coord);
                nnIDXnew(i) = nns_2d_square(nnDict, coord_tmp, coordTransf, incre, K);
                Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord_tmp, nnIDXnew(i), rho, K, 1e-10);
                logliks1(i) = logNormalPdf(Y_i(:), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha + sigma(i)))...
                    -(lambda_sym(1)*norm(invt1Mat0/(t1Mat0), 'fro')+lambda_sym(2)*norm(t1Mat0/invt1Mat0, 'fro'))...
                    +logMVTdistPrior(Tvecprop, priors(i,:), ta0, tb0, lambda);
                
                AP = min(0, logliks1(i)-logliks0(i) + logProposalDiff(Tmat_prop, t1Mat0, diag(diag(Sig(1:6,1:6,i)))));
%                 AP = min(0, logliks1(i)-logliks0(i));

                if (log(rand(1)) < AP)
                    t1(i,:) = Tvecprop(1:6)';
                    accept_i(j) = 1;
                    logliks0(i)=logliks1(i);
                else
                    prop0(j)=0;
                end
            end
        end
        accept(i,:) = accept_i;
        accept_inv(i,:) = accept_inv_i;
    end
end

out.inv_transf_exp = inv_t1;
out.transf_exp = t1;
out.accept = accept;
out.accept_inv = accept_inv;

end


function out = logTdistPrior(Tvec, priors, ta0, tb0, lambda)

Tmat = Vec2Mat(Tvec, false);
PriorMat = Vec2Mat(priors, false);

centered = Tmat*inv(PriorMat);
logCentered = real(logm(centered));

logCenteredVec = [logCentered(:,1); logCentered(:,2)]';

out = sum(log(tpdf(sqrt(lambda*ta0/tb0).*logCenteredVec, 2*ta0)));

end

function out = detADJ(delta)
eigs = eig(adj(delta));
eigs(eigs==0)=[];

out = prod(eigs./(1-exp(-eigs)));
end

function X = adj(A)
[~,n] = size(A);     

[u,s,v] = svd(A);

% compute adjoint of the diagonal matrix s exploiting diagonal property
adjs=zeros(1,n);
for i=1:n
    sdiag=diag(s);
    sdiag(i)=1;
    adjs(i)=prod(sdiag);
end

adjs=diag(adjs); % create a matrix from the diagonal values
X = det(u*v')*v*adjs*u'; % an identity that can be proven by elementary manipulations 
end

function out = logMVNormalPdf(X, MU, Sigma)
R = chol(Sigma);
Z = (X-MU) / R;
out = logNormalPdf(Z, Z*0, 1);
end

function out = logProposalDiff(T1, T0, Sigma)
out = logProposal(T0, T1, Sigma)-logProposal(T1, T0, Sigma);
end

function out = logProposal(T1, T0, Sig)
d1given0 = (T1/T0); % T1*inv(T0)
J = detADJ(d1given0);
out = logMVNormalPdf([d1given0(1:3, 1); d1given0(1:3, 2)]', 0, Sig)+log(abs(1/J));
end
