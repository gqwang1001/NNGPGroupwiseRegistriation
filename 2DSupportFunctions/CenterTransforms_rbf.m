function CenteredOut = CenterTransforms_rbf(out, coord_tmp,Y, KY, nsubj, inverseIndicator, nx, ny, parworkers)

t1_log = out.transf_exp;
% parfor (i = 1:nsubj, parworkers)
for i=1:nsubj
    t1log_mat=logm(Vec2Mat(out.transf_exp(i,:), false));
    t1log_vec = t1log_mat(:);
    t1_log(i,:) = t1log_vec(1:6);
end

% centerize log T

Tlog_mean = mean(t1_log);
Tvec = zeros(nsubj,6);
Tlog_centered = zeros(nsubj,6);

Y_inv = Y;
WKY1 = zeros(size(KY, 1), size(KY, 2), nsubj);

% parfor (i = 1:nsubj, parworkers)
for i=1:nsubj
    Tlog_centered(i,:) = t1_log(i,:) - Tlog_mean;
    if inverseIndicator==true
        Y_inv(:,:,i) = Warp_affine(-Tlog_centered(i,:), Y(:,:,i), true);
    end
    [Tmat,Tvec(i,:)] = my_expm(Tlog_centered(i,:));
    
    WKY1(:,:,i) = warpRBF_affine(Tmat, KY, nx, ny, coord_tmp);
end

CenteredOut.transf_log = Tlog_centered;
CenteredOut.transf_exp = Tvec;
CenteredOut.WKY = WKY1;
CenteredOut.Y_inv = Y_inv;

end

function [tmat_exp, tvec_exp] = my_expm(tvec_log)
tmat = zeros(3);
tmat(:,1) = tvec_log(1:3);
tmat(:,2) = tvec_log(4:6);
tmat_exp = expm(tmat);
tvec_exp9 = tmat_exp(:);
tvec_exp  = tvec_exp9(1:6);
end