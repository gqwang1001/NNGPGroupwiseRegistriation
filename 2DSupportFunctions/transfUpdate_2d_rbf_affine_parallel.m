function out = transfUpdate_2d_rbf_affine_parallel(coord, coord_tmp, transf0, DTs, Y, wKY0, KY, w, sigma, b, ...
    logsig, Sig, nsubj, priors, parworkers, jointUpdate, ta0, tb0, lambda, nx, ny)

logliks0 = zeros(nsubj,1);
logliks1 = zeros(nsubj,1);
% Y_inv = zeros(size(Y,1), size(Y,2), nsubj);
accept = zeros(nsubj, 6);
t1 = transf0; % Tmat

parfor (i=1:nsubj, parworkers)
    
    t1Mat0 = Vec2Mat(t1(i,:), false);
    Tvec = t1Mat0(:);
    Y_i = Y(:,:,i);
    %     Yinv_i = Warp_affine_mat(inv(t1Mat0), Y_i);
    X = cutTemplate(wKY0(:,:,i)*w, coord, coord_tmp);
    logliks0(i) = logNormalPdf(Y_i(:), b(i)*X, sqrt(+sigma(i)))+...
        logMVTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda);
end

% MH updates
if jointUpdate == true
    parfor (i = 1:nsubj, parworkers)
        Y_i = Y(:,:,i);
        prop0 = my_mvnrnd(zeros(1, 6), exp(logsig(i,1))*Sig(1:6,1:6,i));
        Tmat_prop = expm(Vec2Mat(prop0, true))*DTs(:,:,i)*Vec2Mat(t1(i,:), false);
        if ~checkLogMat(Tmat_prop) && ~checkLogMat(Tmat_prop*inv(Vec2Mat(priors(i,:), false)))
            Tvec = Tmat_prop(:);
            %
            wKY1 = warpRBF_affine(Tmat_prop, KY, nx, ny, coord_tmp);
            X = cutTemplate(wKY1*w, coord, coord_tmp);
            
            %         Yinv_i1 = Warp_affine(-Tlog_prop, Y(:,:,i));
            logliks1(i) = logNormalPdf(Y_i(:), b(i)*X, sqrt(sigma(i)))+...
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
    
% else
%     
%     parfor (i=1:nsubj, parworkers)
%         %     for i=1:nsubj
%         Y_i = Y(:,:,i);
%         
%         for j = 1:6
%             t1Mat0 = Vec2Mat(t1(i,:), false);
%             prop0 = zeros(1, 6);
%             
%             prop0(j) = exp(logsig(i,j))/2*DTs(i,j)+randn(1)*sqrt(exp(logsig(i,j))*Sig(j,j,i));
%             Tmat_prop = expm(Vec2Mat(prop0, true))*t1Mat0;
%             
%             if ~checkLogMat(Tmat_prop) && ~checkLogMat(Tmat_prop*inv(Vec2Mat(priors(i,:), false)))
%                 Tvec = Tmat_prop(:);
%                 wKY1 = warpRBF_affine(Tmat_prop, KY, nx, ny);
%                 %         Yinv_i1 = Warp_affine(-Tlog_prop, Y(:,:,i));
%                 logliks1(i) = logNormalPdf(Y_i(:), b(i)*wKY1*w, sqrt(sigma(i)))+...
%                     logTdistPrior(Tvec, priors(i,:), ta0, tb0, lambda);
%                 
%                 AP = min(0, logliks1(i)-logliks0(i));
%                 if (log(rand(1)) < AP)
%                     t1(i,:) = Tvec(1:6)';
%                     accept(i,j) = 1;
%                     logliks0(i) = logliks1(i);
%                 end
%             end
%         end
%     end
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

function out = my_expm(tvec_log)
tmat = zeros(3);
tmat(:,1) = tvec_log(1:3);
tmat(:,2) = tvec_log(4:6);
tmat_exp = expm(tmat);
tmat1 = tmat_exp(:,1:2);
out = tmat1(:);
end

function out = logTdistPrior(Tvec, priors, ta0, tb0, lambda)

Tmat = Vec2Mat(Tvec, false);
PriorMat = Vec2Mat(priors, false);

centered = Tmat*inv(PriorMat);
logCentered = real(logm(centered));

logCenteredVec = [logCentered(:,1); logCentered(:,2)]';

out = sum(log(tpdf(sqrt(lambda*ta0/tb0).*logCenteredVec, 2*ta0)));

end