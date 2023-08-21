function out = Transform_coord_affine(tvec_log, coord)

    tmat = zeros(3);
    tmat(:,1) = tvec_log(1:3);
    tmat(:,2) = tvec_log(4:6);

    tmat_exp = expm(tmat);

    out3 = [coord, ones(size(coord, 1),1)]*inv(tmat_exp);
    out = out3(:,1:2);
end