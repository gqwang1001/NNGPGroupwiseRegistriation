function CenteredOut = CenterTransforms(out, Y, nnIDXs, nsubj, coord, incre, K, nnDict, inverseIndicator)

t1_log = out.transf_exp;
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

for i = 1:nsubj
    Tlog_centered(i,:) = t1_log(i,:) - Tlog_mean;
    if inverseIndicator==true
        Y_inv(:,:,i) = Warp_affine(-Tlog_centered(i,:), Y(:,:,i), true);
    end
    Tvec(i,:) = my_expm(Tlog_centered(i,:));
    
    coordTransfs = Transform_coord_affine(Tlog_centered(i,:), coord);
    nnIDXs(i) = nns_2d_square(nnDict, coord, coordTransfs, incre, K);
end

CenteredOut.transf_log = Tlog_centered;
CenteredOut.transf_exp = Tvec;
CenteredOut.nnIDXs = nnIDXs;
CenteredOut.Y_inv = Y_inv;

end

function tvec_exp = my_expm(tvec_log)
tmat = zeros(3);
tmat(:,1) = tvec_log(1:3);
tmat(:,2) = tvec_log(4:6);
tmat_exp = expm(tmat);
tvec_exp9 = tmat_exp(:);
tvec_exp  = tvec_exp9(1:6);
end