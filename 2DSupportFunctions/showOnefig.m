function showOnefig(Y, i, iter)

[ncoord] = length(Y);
Ymat = zeros(sqrt(ncoord), sqrt(ncoord));

figure(i);
Ymat(:) = Y;
imagesc(Ymat); colormap jet; colorbar;
title(['Iteration ', num2str(iter)]);
end
