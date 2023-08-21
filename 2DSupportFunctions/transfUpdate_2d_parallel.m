function out = transfUpdate_2d_parallel(transf0, nnIDXs, X, Y, sigma, alpha, rho, coord,K, b, nnDict, logsig, Sig, Cov_NN0, incre, nsubj, bnds, priors, parworkers)

logliks0 = zeros(nsubj,1);
logliks1 = zeros(nsubj,1);
A  = zeros(nsubj,1);
accept = zeros(nsubj, 5);
nparams = size(transf0, 2);

t1 = transf0;
nnIDXnew = nnIDXs;
Covs_NN1 = Cov_NN0;

parfor (i = 1:nsubj, parworkers)
    
    tOut = t1(i,:);
    accept_i = zeros(1, nparams);
    
    
    % scaling X
    logliks0(i) = logNormalPdf(Y(:,i), b(i)*Cov_NN0(i).mu, sqrt(b(i)^2*Cov_NN0(i).Ft * alpha + sigma(i)));
    tfLogProp = randn(1) * sqrt(exp(logsig(i,1))*Sig(1,1,i)) + log(tOut(1));
    
%     if exp(tfLogProp)>bnds(1,1) && exp(tfLogProp)<bnds(1,2)
        
        tf1Prop = [exp(tfLogProp), tOut(2:5)];
        
        coordTransf = Transfer_Coord(tf1Prop(1:2), tf1Prop(3), tf1Prop(4:5), coord);
        nnIDXnew(i) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
        Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord, nnIDXnew(i), rho, K, 1e-10);
        logliks1(i) = logNormalPdf(Y(:,i), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha +sigma(i)));
        
        A(i) = min(0, logliks1(i)-logliks0(i) +...
            log(normpdf(log(tf1Prop(1)), 0, priors(1)))- log(normpdf(log(tOut(1)), 0, priors(1))));
        
        if (log(rand(1)) < A(i))
            tOut = tf1Prop;
            accept_i(1)  = 1;
            logliks0(i) = logliks1(i);
        end
%     end
    % scaling Y
    tfLogProp = randn(1) * sqrt(exp(logsig(i,2))*Sig(2,2,i)) + log(tOut(2));
    
%     if exp(tfLogProp)>bnds(1,1) && exp(tfLogProp)<bnds(1,2)
        
        tf1Prop = [tOut(1) exp(tfLogProp) tOut(3:5)];
        
        coordTransf = Transfer_Coord(tf1Prop(1:2), tf1Prop(3), tf1Prop(4:5), coord);
        nnIDXnew(i) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
        Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord, nnIDXnew(i), rho, K, 1e-10);
        logliks1(i) = logNormalPdf(Y(:,i), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha +sigma(i)));
        A(i) = min(0, logliks1(i)-logliks0(i) +...
            log(normpdf(log(tf1Prop(2)), 0, priors(1)))- log(normpdf(log(tOut(2)), 0, priors(1))));
        
        if (log(rand(1)) < A(i))
            tOut = tf1Prop;
            accept_i(2)  = 1;
            logliks0(i) = logliks1(i);
            %             Covs_NN0(i) = Covs_NN1(i);
        end
        
%     end
    
    %         rotation : truncated normal
    %         pdata = randn(500,1) * sqrt(exp(logsig(i,3))*Sig(3,3,i));
    %         pd = fitdist(pdata, "Normal");
    %         pd10 = truncate(pd, -pi/2 - tOut(3)/180*pi, pi/2 - tOut(3)/180*pi);
    %         tfProp = tOut(3)/180*pi + random(pd10, 1);
    %         logProb10 = log(pdf(pd10, tfProp - tOut(3)/180*pi));
    
    %         pd01 = truncate(pd, -pi/2 - tfProp/180*pi, pi/2 - tfProp/180*pi);
    %         logProb01 = log(pdf(pd01, tOut(3) - tfProp/180*pi));
    
    
    tfProp = randn(1) * sqrt(exp(logsig(i,3))*Sig(3,3,i)) + tOut(3)/180*pi;
    %
%     if (tfProp/pi*180)>bnds(2,1) && (tfProp/pi*180)<bnds(2,2)
        
        tf1Prop = [tOut(1:2), tfProp/pi*180, tOut(4:5)];
        
        coordTransf = Transfer_Coord(tf1Prop(1:2), tf1Prop(3), tf1Prop(4:5), coord);
        nnIDXnew(i) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
        Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord, nnIDXnew(i), rho, K, 1e-10);
        logliks1(i) = logNormalPdf(Y(:,i), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha +sigma(i)));
        A(i) = min(0, logliks1(i)-logliks0(i)+...
            log(normpdf(tf1Prop(3)/180*pi, 0, priors(2)))- log(normpdf(tOut(3)/180*pi, 0, priors(2))));
        if (log(rand(1)) < A(i))
            tOut = tf1Prop;
            accept_i(3)  = 1;
            logliks0(i) = logliks1(i);
            %             Covs_NN0(i) = Covs_NN1(i);
        end
%     end
    
    % shifting X
    tfProp = randn(1) * sqrt(exp(logsig(i,4))*Sig(4,4,i)) + tOut(4);
    
%     if tfProp>bnds(3,1) && tfProp<bnds(3,2)
        
        tf1Prop = [tOut(1:3) tfProp tOut(5)];
        coordTransf = Transfer_Coord(tf1Prop(1:2), tf1Prop(3), tf1Prop(4:5), coord);
        nnIDXnew(i) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
        Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord, nnIDXnew(i), rho, K, 1e-10);
        logliks1(i) = logNormalPdf(Y(:,i), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha +sigma(i)));
        A(i) = min(0, logliks1(i)-logliks0(i)+...
            log(normpdf(tf1Prop(4), 0, priors(3)))- log(normpdf(tOut(4), 0, priors(3))));
        
        if (log(rand(1)) < A(i))
            tOut = tf1Prop;
            accept_i(4)  = 1;
            logliks0(i) = logliks1(i);
            %             Covs_NN0(i) = Covs_NN1(i);
        end
%     end
    % shifting Y        
    tfProp = randn(1) * sqrt(exp(logsig(i,5))*Sig(5,5,i)) + tOut(5);

%     if tfProp>bnds(3,1) && tfProp<bnds(3,2)
        
        tf1Prop = [tOut(1:4) tfProp];
        coordTransf = Transfer_Coord(tf1Prop(1:2), tf1Prop(3), tf1Prop(4:5), coord);
        nnIDXnew(i) = nns_2d_square(nnDict, coord, coordTransf, incre, K);
        Covs_NN1(i) = Cov_NN_Transfer_2d(X, coord, nnIDXnew(i), rho, K, 1e-10);
        logliks1(i) = logNormalPdf(Y(:,i), b(i)*Covs_NN1(i).mu, sqrt(b(i)^2*Covs_NN1(i).Ft * alpha +sigma(i)));
        A(i) = min(0, logliks1(i)-logliks0(i)+...
            log(normpdf(tf1Prop(5), 0, priors(3)))- log(normpdf(tOut(5), 0, priors(3))));
        
        if (log(rand(1)) < A(i))
            tOut = tf1Prop;
            accept_i(5)  = 1;
            logliks0(i) = logliks1(i);
            %             Covs_NN0(i) = Covs_NN1(i);
        end
%     end
    t1(i,:) = tOut;
    accept(i, :) = accept_i;
    
end

% reject if any scaling is negative
%     tCentered = [exp(log(t1(:,1))-mean(log(t1(:,1)))), t1(:,2)-mean(t1(:,2))];
tCentered = [t1(:,1)./ mean(t1(:,1)),...
     t1(:,2)./ mean(t1(:,2)),...
    (t1(:,3) - mean(t1(:,3))),...
    (t1(:,4) - mean(t1(:,4))),...
    (t1(:,5) - mean(t1(:,5)))];

tOut = transf0;
acceptOut = zeros(nsubj, 5);


tTreshed_scalingX = tCentered(:,1)<bnds(1,1) | tCentered(:,1)>bnds(1,2);
tTreshed_scalingY = tCentered(:,2)<bnds(1,1) | tCentered(:,2)>bnds(1,2);
tTreshed_rotation = tCentered(:,3)<bnds(2,1) | tCentered(:,3)>bnds(2,2);
tTreshed_shiftX   = tCentered(:,4)<bnds(3,1) | tCentered(:,4)>bnds(3,2);
tTreshed_shiftY   = tCentered(:,5)<bnds(3,1) | tCentered(:,5)>bnds(3,2);
       
if sum(tTreshed_scalingX+tTreshed_scalingY+tTreshed_rotation+tTreshed_shiftX+tTreshed_shiftY) == 0
    tOut = tCentered;
    for i = 1:nsubj
        coordTransfs = Transfer_Coord(tOut(i,1:2), tOut(i,3), tOut(i,4:5), coord);
        nnIDXs(i) = nns_2d_square(nnDict, coord, coordTransfs, incre, K);
    end
    acceptOut = accept;
end

out.transf = tOut;
out.nnIDXs = nnIDXs;
out.accept = acceptOut;
end