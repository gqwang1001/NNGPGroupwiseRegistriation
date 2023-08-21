function out = my_expm(tvec_log)
tmat = zeros(3);
tmat(:,1) = tvec_log(1:3);
tmat(:,2) = tvec_log(4:6);
tmat_exp = expm(tmat);
tmat1 = tmat_exp(:,1:2);
out = tmat1(:);
end