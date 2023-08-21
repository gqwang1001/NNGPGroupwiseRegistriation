function showShifts(tMat_exp, fg, iter)
f = figure(fg);
f.Position = [1500,100,1500, 1000];
nsubj = size(tMat_exp, 2);

for i = 1:nsubj
    subplot(6,6, i);
    plot(1:iter, tMat_exp(1:iter, i, 3));
    title(['b1 ', num2str(i)]);
end