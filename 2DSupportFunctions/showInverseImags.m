function showInverseImags(Y_inv,fg)
f = figure(fg);
nsubj = size(Y_inv, 3);
f.Position = [1500,100,1500, 1000];
for i = 1:nsubj
subplot(6,6,i);
imagesc(Y_inv(:,:,i)); 
colormap jet; colorbar; title(i);
end
end