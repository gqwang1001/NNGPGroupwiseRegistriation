function out = Transform_mat_coord_affine(tmat, coord)

    out3 = [coord, ones(size(coord, 1),1)]*inv(tmat);
    out = out3(:,1:2);
    
end