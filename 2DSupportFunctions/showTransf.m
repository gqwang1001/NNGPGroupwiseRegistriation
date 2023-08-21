function showTransf(tMat_exp, fg, iter)
f = figure(fg);
f.Position = [2500,100,1000, 1000];
for i = 1:3
    subplot(6, 3, i);
    plot(1:iter, tMat_exp(1:iter, i, 1));
    title(['A11 Subj', num2str(i)]);
    
    subplot(6, 3, i+3);
    plot(1:iter,tMat_exp(1:iter, i, 2));
    title(['A21 Subj', num2str(i)]);    
    
    subplot(6, 3, i+6);
    plot(1:iter, tMat_exp(1:iter, i, 4));
    title(['A12 Subj', num2str(i)]);
    
    subplot(6, 3, i+9);
    plot(1:iter, tMat_exp(1:iter, i, 5));
    title(['A22 Subj', num2str(i)]);    
    
    subplot(6, 3, i+12);
    plot(1:iter, tMat_exp(1:iter, i, 3));
    title(['b1 Subj', num2str(i)]);
    
    subplot(6, 3, i+15);
    plot(1:iter, tMat_exp(1:iter, i, 6));
    title(['b2 Subj', num2str(i)]);
end