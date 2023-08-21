function CenteredOut = sym_CenterTransforms_largeX(out, Y, nnIDXs, nsubj, coord, coord_tmp, incre, K, nnDict)

Tout = centerTransf(out.transf_exp, nsubj);
Tinvout = centerTransf(out.inv_transf_exp, nsubj);

Y_inv = Y;
for i = 1:nsubj
    Y_inv(:,:,i) = Warp_affine_mat(Tinvout.Texp(:,:,i), Y(:,:,i));
    coordTransfs = Transform_mat_coord_affine(Tout.Texp(:,:,i), coord);
    nnIDXs(i) = nns_2d_square(nnDict, coord_tmp, coordTransfs, incre, K);
end

CenteredOut.transf_log = Tout.Tlog_centered;
CenteredOut.transf_exp = Tout.Tvec;
CenteredOut.invtransf_log = Tinvout.Tlog_centered;
CenteredOut.invtransf_exp = Tinvout.Tvec;
CenteredOut.nnIDXs = nnIDXs;
CenteredOut.Y_inv = Y_inv;
end


function out = centerTransf(transf_exp, nsubj)
t1_log_A = zeros(2,2,nsubj);

for i=1:nsubj
    t1_log_A(:,:,i)=logm(reshape(transf_exp(i,[1,2,4,5])', 2, 2));
end

% centerize log T
Tlog_A_mean = mean(t1_log_A, 3);
Tlog_shift_mean = mean(transf_exp(:,[3,6]));
Tlog_centered_A = t1_log_A-repmat(Tlog_A_mean, 1,1,nsubj);
Tlog_centered_shift = transf_exp(:,[3,6]) - repmat(Tlog_shift_mean, nsubj,1);

Texp = zeros(3,3,nsubj);
Tvec = zeros(nsubj,6);
Tlog_centered = zeros(nsubj,6);

for i = 1:nsubj
    Texp(:,:,i) = [expm(Tlog_centered_A(:,:,i)), [0;0]; Tlog_centered_shift(i,:), 1];
    Tvec(i,:) = [Texp(:,1,i);Texp(:,2,i)]';
    Tlog_centered(i,:) = reshape([Tlog_centered_A(:,:,i);Tlog_centered_shift(i,:)], [],1);
end
out.Tvec = Tvec;
out.Texp = Texp;
out.Tlog_centered = Tlog_centered;
end
