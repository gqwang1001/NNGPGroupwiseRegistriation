function showYinv(Y_inv, fg)
f = figure(fg);
f.Position = [0,100, 3e3/2, 1e3/3];
for i = 1:3
    subplot(1, 3, i);
    imagesc(Y_inv(:,:,i));
    colormap jet;
    colorbar;
end