function out = transfUpdate_2d_largeX_affine_parallel(transf0, DTs, nnIDXs,...
    X, Y, sigma, alpha, rho, coord, coord_tmp, K, b, nnDict, logsig, Sig, ...
    Cov_NN0, incre, nsubj, bnds, priors, parworkers, jointUpdate, ...
    ta0, tb0, lambda)

logliks0 = zeros(nsubj,1);
logliks1 = zeros(nsubj,1);
% Y_inv = zeros(size(Y,1), size(Y,2), nsubj);
accept = zeros(nsubj, 6);
t1 = transf0; % Tmat
nnIDXnew = nnIDXs;
Covs_NN1 = Cov_NN0;

parfor (i=1:nsubj, parworkers)
    % for i=1:nsubj

    t1Mat0 = Vec2Mat(t1(i,:), false);
    Tvec = t1Mat0(:);
    Y_i = Y(:,:,i);
    %     Yinv_i = Warp_affine_mat(inv(t1Mat0), Y_i);
    %     logliks0(i) = sym_logNormalPdf(Yinv_i(:), b(i)*X, sqrt(sigma(i)), Y_i(:), b(i)*Cov_NN0(i).mu, sqrt(b(i)^2*Cov_NN0(i).Ft * alpha + sigma(i)))+...
    %             sum(log(normpdf(Tvec(1:6)', priors(i,:), lambda)));
    logliks0(i) = logNormalPdf(Y_i(:), b(i)*Cov_NN0(i).mu, sqrt(b(i)^2*Cov_NN0(i).Ft * alpha +sigma(i)))+...
        logMVTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda);
end

% MH updates
if jointUpdate == true

%     parfor (i=1:nsubj, parworkers)
    for i=1:nsubj
        Y_i = Y(:,:,i);
        %         prop0 = my_mvnrnd(exp(logsig(i,1))/2*DTs(i,:), exp(logsig(i,1))*Sig(1:6,1:6,i));
        prop0 = my_mvnrnd(zeros(1, 6), exp(logsig(i,1))*Sig(1:6,1:6,i));

        Tmat_prop = my_expm(prop0)*DTs(:,:,i)*Vec2Mat(t1(i,:), false);

        if ~checkLogMat(Tmat_prop) && ~checkLogMat(Tmat_prop*inv(Vec2Mat(priors(i,:), false)))

            Tvec = Tmat_prop(:);

            coordTransf = Transform_mat_coord_affine(Tmat_prop, coord);
            nnIDXnew(i) = nns_2d_square(nnDict, coord_tmp, coordTransf, incre, K);
            Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord_tmp, nnIDXnew(i), rho, K, 1e-10);
            %         Yinv_i1 = Warp_affine(-Tlog_prop, Y(:,:,i));
            logliks1(i) = logNormalPdf(Y_i(:), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha + sigma(i)))+...
                logMVTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda);
            %         logliks1(i) = sym_logNormalPdf(Yinv_i1(:), b(i)*X, sqrt(sigma(i)), Y_i(:), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha + sigma(i)))+...
            %             sum(log(tpdf(sqrt(lambda*ta0/tb0).*(Tlog_prop - priors(i,:)), 2*ta0)));
            %     sum(log(normpdf(Tlog_prop, priors(i,:), lambda)));

            AP = min(0, logliks1(i)-logliks0(i));
            if (log(rand(1)) < AP)
                t1(i,:) = Tvec(1:6)';
                accept(i,1) = 1;
                %             Y_inv(:,:,i) = Yinv_i1;
            end
        end
    end

else

%     parfor (i=1:nsubj, parworkers)
    for i=1:nsubj
        Y_i = Y(:,:,i);

        for j = 1:6
            t1Mat0 = Vec2Mat(t1(i,:), false);
            prop0 = zeros(1, 6);

            %             prop0(j) = exp(logsig(i,j))/2*DTs(i,j)+randn(1)*sqrt(exp(logsig(i,j))*Sig(j,j,i));
            Tmat_prop = expm(Vec2Mat(prop0, true))*t1Mat0;

            if ~checkLogMat(Tmat_prop) && ~checkLogMat(Tmat_prop*inv(Vec2Mat(priors(i,:), false)))
                Tvec = Tmat_prop(:);
                %             if Tlog_prop(j)>bnds(j,1) && Tlog_prop(j)<bnds(j,2)
                coordTransf = Transform_mat_coord_affine(Tmat_prop, coord);
                nnIDXnew(i) = nns_2d_square(nnDict, coord_tmp, coordTransf, incre, K);
                Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord_tmp, nnIDXnew(i), rho, K, 1e-10);
                %             Yinv_i1 = Warp_affine_mat(inv(Tmat_prop), Y_i);
                %             logliks1(i) = sym_logNormalPdf(Yinv_i1(:), b(i)*X, sqrt(sigma(i)), Y_i(:), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha + sigma(i)))+...
                %                                         sum(log(normpdf(Tvec(1:6)', priors(i,:), lambda)));
                logliks1(i) = logNormalPdf(Y_i(:), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha + sigma(i)))+...
                    logMVTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda);

                AP = min(0, logliks1(i)-logliks0(i));
                if (log(rand(1)) < AP)
                    t1(i,:) = Tvec(1:6)';
                    accept(i,j) = 1;
                    logliks0(i) = logliks1(i);
                end
            end
        end
    end
end

acceptOut = accept;
out.transf_exp = t1;
out.accept = acceptOut;

end

function out = my_mvnrnd(mu, Sig)
n = length(mu);
R = chol(Sig);
out = mu + randn(1, n) * R;
end

function tmat = my_expm(tvec_log)
tmat = eye(3);
tmat(1:2, 1:2) = expm([tvec_log(1:2);tvec_log(4:5)]');
tmat(3, 1:2) = tvec_log([3 6]);
end

function out = logTdistPrior(Tvec, priors, ta0, tb0, lambda)

Tmat = Vec2Mat(Tvec, false);
PriorMat = Vec2Mat(priors, false);

centered = Tmat*inv(PriorMat);
logCentered = real(logm(centered));

logCenteredVec = [logCentered(:,1); logCentered(:,2)]';

out = sum(log(tpdf(sqrt(lambda*ta0/tb0).*logCenteredVec, 2*ta0)));

end


