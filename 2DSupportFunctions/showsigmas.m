function showsigmas(b, sigma, fg, iter)

f = figure(fg);
f.Position = [500,100,1000, 1000];
for i = 1:3
    subplot(3, 3, i);
    plot(1:iter,b(1:iter, i));
    title(['Beta Subj', num2str(i)]);
    
    subplot(3, 3, i+3);
    plot(1:iter,sigma(1:iter, i));
    title(['Sigma Subj', num2str(i)]);    
end
end