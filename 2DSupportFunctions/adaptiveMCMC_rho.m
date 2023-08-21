function logsigRho = adaptiveMCMC_rho(adapt, iter, k, acceptMatRho, c0, c1, ropt, logsigRho)

    if (adapt && rem(iter, k)==0)
        rhatRho = mean(acceptMatRho((iter-k+1):iter));
        gamma1 = 1/((floor(iter/k)+1)^c1);
        gamma2 = c0*gamma1;
        logsigRho = logsigRho+gamma2*(rhatRho-ropt);
    end
end