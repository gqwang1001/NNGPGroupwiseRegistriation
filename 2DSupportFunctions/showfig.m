function showfig(Y)

[ncoord, nsubj] = size(Y);
Ymat = zeros(sqrt(ncoord), sqrt(ncoord));

figure;
for i = 1:nsubj
    Ymat(:) = Y(:,i);
    subplot(1, nsubj, i);
    imagesc(Ymat); colormap jet; colorbar;
end
end



