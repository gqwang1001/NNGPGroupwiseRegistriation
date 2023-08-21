function mat = Vec2MatA(v, logIndicator)

if logIndicator == true
    mat = zeros(3);
else
    mat = eye(3);
end

    mat(1:3, 1) = v(1:3);
    mat(1:3, 2) = v(4:6);
end

