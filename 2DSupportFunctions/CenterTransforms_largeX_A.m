function CenteredOut = CenterTransforms_largeX_A(out, Y, nnIDXs, nsubj, coord, coord_tmp, incre, K, nnDict, inverseIndicator)

t1_log_A = zeros(2,2,nsubj);

for i=1:nsubj
    t1_log_A(:,:,i)=logm(reshape(out.transf_exp(i,[1,2,4,5])', 2, 2));
end

% centerize log T
Tlog_A_mean = mean(t1_log_A, 3);
Tlog_shift_mean = mean(out.transf_exp(:,[3,6]));
Tlog_centered_A = t1_log_A-repmat(Tlog_A_mean, 1,1,nsubj);
Tlog_centered_shift = out.transf_exp(:,[3,6]) - repmat(Tlog_shift_mean, nsubj,1);

Tvec = zeros(nsubj,6);
Tlog_centered = zeros(nsubj,6);

Y_inv = Y;

for i = 1:nsubj
    
    Texp = [expm(Tlog_centered_A(:,:,i)), [0;0]; Tlog_centered_shift(i,:), 1];
    Tvec(i,:) = [Texp(:,1);Texp(:,2)]';
    Tlog_centered(i,:) = reshape([Tlog_centered_A(:,:,i);Tlog_centered_shift(i,:)], [],1);
    coordTransfs = Transform_mat_coord_affine(Texp, coord);
    nnIDXs(i) = nns_2d_square(nnDict, coord_tmp, coordTransfs, incre, K);
    if inverseIndicator==true
        Y_inv(:,:,i) = Warp_affine_mat(inv(Texp), Y(:,:,i));
    end
end

CenteredOut.transf_log = Tlog_centered;
CenteredOut.transf_exp = Tvec;
CenteredOut.nnIDXs = nnIDXs;
CenteredOut.Y_inv = Y_inv;

end