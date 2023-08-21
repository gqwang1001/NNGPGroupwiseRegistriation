function showSummaryFigs(XMat, i, iter, k)

ncoord = size(XMat,2);

ranges = iter-(1:k)+1;
LatentImg = ones(sqrt(ncoord),sqrt(ncoord));
LatentImg(:) = mean(XMat(ranges, :));
sdImg = ones(size(LatentImg));
sdImg(:) = std(XMat(ranges, :), 0, 1);

f = figure(i);
f.Position=[0,500,3e3/2, 1e3/3];
title(['Iteration', num2str(iter)]);
subplot(1,3, 1);imagesc(LatentImg); colormap jet; colorbar; title('Posterior Mean');
subplot(1,3, 2);imagesc(sdImg); colormap jet; colorbar;title('Posterior SD');
subplot(1,3, 3);imagesc(LatentImg./sdImg); colormap jet; colorbar;title('Posterior T-stat');
end

