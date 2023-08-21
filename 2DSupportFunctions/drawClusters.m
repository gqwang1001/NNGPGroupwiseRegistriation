function drawClusters(BW1, TStatMap, titleText)

% [x1, y1] = find(BW1==1);
% k = boundary(x1, y1);

figure; 
imagesc(TStatMap); colormap jet; colorbar; title(titleText);
hold on;
visboundaries(BW1, 'Color','k');