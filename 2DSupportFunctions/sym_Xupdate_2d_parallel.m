function Xnew = sym_Xupdate_2d_parallel(YT_inv, bi, sigma, XT, XOld, nnIDXs, nnX, nsubj, alpha, Cov_NN, coord, rho, parworkers)

    % inverse-reg part
    F1 = 1/sum(bi.^2./sigma);
    B1mat = reshape(YT_inv, [], nsubj);
    for subj = 1:nsubj
        Yinvmat = YT_inv(:,:,subj);
        YinvVec = Yinvmat(:);
        B1mat(:,subj) = YinvVec*bi(subj)/sigma(subj);
    end
    B1 = sum(B1mat, 2);
    
    
    Xnew = XOld;
%     expDists = exp(-rho * nnX.Dists);
    BsFsX = CondLatentX_2d(Xnew, coord, rho, nnX, false,1e-9);
    
    for i = 1:length(XOld)
        
       BFa1 = 0;
       BFB1 = 0;
       if i==1
           FBXN = 0;
           Fs = alpha;
       else
           
        idxs = nnX.Idxs(i, :);
        iidxs = nnX.Idxs(i, :)>0;
        idxs = idxs(iidxs);
        
%           B_s =  expDists(i, idxs) / exp(-rho * squareform(pdist(coord(idxs))));
          Fs  = alpha*BsFsX.Fs(i);
          FBXN = 1/Fs * dot(BsFsX.Bs(i, iidxs), Xnew(idxs));
          
         % case 1 : within the latent map
          [nnset_r, nnset_c] = find(nnX.Idxs == i); 
          
          if ~isempty(nnset_r)
              for t = 1:length(nnset_r)
                  
                  idxt = nnX.Idxs(nnset_r(t), :);
                  btsi = BsFsX.Bs(nnset_r(t), nnset_c(t));
                  Ft = alpha*BsFsX.Fs(nnset_r(t));
                  
                  atsIdx = (idxt ~= i & idxt > 0); 
                  ats = Xnew(nnset_r(t)) - dot(BsFsX.Bs(nnset_r(t), atsIdx), Xnew(nnX.Idxs(nnset_r(t), atsIdx)));
                  
                  BFa1 = BFa1 + btsi/Ft * ats;
                  BFB1 = BFB1 + btsi/Ft * btsi;
              end
          end
       end
       
          % case 2 : conditional on X(T_i)
          
          BFa2 = 0;
          BFB2 = 0;
          parfor (subj = 1:nsubj, parworkers)
              nnTi = nnIDXs(subj);          
              [nnset_r, nnset_c] = find(nnTi.Idxs == i); 
              if ~isempty(nnset_r)
                  Bt = Cov_NN(subj).Bt;
                  for t = 1:length(nnset_r)
                      btsi = Bt(nnset_r(t), nnset_c(t));
                      atsIdx = nnTi.Idxs(nnset_r(t),:) ~=i;
                      ats = XT(nnset_r(t),subj) - dot(Bt(nnset_r(t), atsIdx), Xnew(nnTi.Idxs(nnset_r(t), atsIdx)));
                      
                      BFa2 = BFa2 + btsi/(alpha * Cov_NN(subj).Ft(nnset_r(t))) * ats;
                      BFB2 = BFB2 + btsi/(alpha * Cov_NN(subj).Ft(nnset_r(t))) * btsi;
                  end
              end
          end
              
          Vi = 1/(1/Fs + BFB1 + BFB2 + 1/F1);
          Mui = FBXN + BFa1 + BFa2 + B1(i);
          Xnew(i) = sqrt(Vi) * randn(1) + Mui * Vi;
    end

end

