function [logsig, Sig] = adaptiveMCMC(adapt, iter, k, tstart, tMat, acceptMat, logsig, Sig, c0, c1, ropt, nsubj, npar)

rhat = zeros(nsubj, npar);
if (adapt && rem(iter, k)==0)
%     disp(iter);
%     disp(toc(tstart));
    for subj = 1:nsubj
        Sig0tHat = cov(squeeze(tMat((iter-k+1):iter,subj,:)));
        for j = 1:npar
            rhat(subj, j) = mean(acceptMat((iter-k+1):iter, subj, j));
            gamma1 = 1/((floor(iter/k)+1)^c1);
            gamma2 = c0*gamma1;
            
            logsig(subj, j) = logsig(subj, j) + gamma2 *(rhat(subj, j)-ropt);
        end
        Sig(:,:,subj) = Sig(:,:,subj)+gamma1*(Sig0tHat-Sig(:,:,subj));
    end
end

end