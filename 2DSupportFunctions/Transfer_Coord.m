function coord_trans = Transfer_Coord(scaling, rotation, translation, coord)

tmat = [cosd(rotation), sind(rotation), 0;...
       -sind(rotation), cosd(rotation), 0;...
        0 0 1] * ...
       diag([scaling(1), scaling(2), 1]);
tmat(3, 1) = translation(1);
tmat(3, 2) = translation(2);

coord_tf = [coord, ones(size(coord,1),1)] * tmat;
coord_trans = coord_tf(:, 1:2);

end

