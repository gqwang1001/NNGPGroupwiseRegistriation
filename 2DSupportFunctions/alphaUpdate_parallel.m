function out = alphaUpdate_parallel(Xlatent, XT, nsubj, Cov_NN,muFsLatentX, sa0, sb0, parworkers)
    sse1 = zeros(nsubj, 1);
    
    parfor (subj = 1:nsubj, parworkers)
        sse1(subj) = sum((XT(:,subj) - Cov_NN(subj).mu).^2 ./ Cov_NN(subj).Ft);
    end
    
    sse2 = sum((Xlatent - muFsLatentX.mu).^2 ./ (muFsLatentX.Fs));
    out = 1/gamrnd(sa0 + double(0.5 * (size(XT,2)*size(XT,1)+length(Xlatent))), 1./(sb0 + 0.5 * (sum(sse1) + sse2)));

end

