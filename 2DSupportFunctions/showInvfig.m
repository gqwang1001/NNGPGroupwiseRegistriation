function showInvfig(Y)
nsubj = size(Y, 3);
figure;
for i = 1:nsubj
    subplot(1, nsubj, i);
    imagesc(Y(:,:,i)); colormap jet; colorbar;
end
end