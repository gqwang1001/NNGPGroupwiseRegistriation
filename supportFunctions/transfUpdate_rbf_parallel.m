function out = transfUpdate_rbf_parallel(transf0, Y, sigma, ...
    wKY0,KY,w, coord, b, logsig, Sig, nsubj, parworkers)

eps=0;

logliks0 = zeros(nsubj,1);
logliks1 = zeros(nsubj,1);
A  = zeros(nsubj,1);
accept = zeros(nsubj, 2);

tf0Log = transf0;
tf0Log(:,1) = log(tf0Log(:,1));

t1 = transf0;
nparams = size(t1, 2);
% nnIDXnew = nnIDXs;
% Covs_NN1 = Cov_NN0;

% parfor (i = 1:nsubj, parworkers)
for i=1:nsubj
    tOut_i = t1(i, :);
    accept_i = zeros(1, nparams);
    logliks0(i) = logNormalPdf(Y(:,i), b(i)*wKY0(:,:,i)*w, sqrt(sigma(i)));
    
    tf1LogProp = randn(1) * sqrt(exp(logsig(i,1))*Sig(1,1,i)) + log(tOut_i(1));
    if abs(tf1LogProp)<1
        tf1Prop = [exp(tf1LogProp) tOut_i(2)];
        coordTransf = tf1Prop * [coord ones(length(coord),1)]';
        wKY1 = warpRBF_1d(coord, coordTransf, KY);
        logliks1(i) = logNormalPdf(Y(:,i), b(i)*wKY1*w, sqrt(sigma(i)));
        
        A(i) = min(0, logliks1(i)-logliks0(i)+...
            log(normpdf(log(tf1Prop(1)), 0, .1))- log(normpdf(log(tOut_i(1)), 0, .1)));
        if (log(rand(1)) < A(i))
            tOut_i = tf1Prop;
            accept_i(1)  = 1;
            logliks0(i) = logliks1(i);
        end
    end
    % shifting parameters
    tf1LogProp = randn * sqrt(exp(logsig(i,2))*Sig(2,2,i)) + tOut_i(2);
    tf1Prop = [tOut_i(1) tf1LogProp];
    
    coordTransf = tf1Prop * [coord ones(length(coord),1)]';
    wKY1 = warpRBF_1d(coord, coordTransf, KY);
    logliks1(i) = logNormalPdf(Y(:,i), b(i)*wKY1*w, sqrt(sigma(i)));
    A(i) = min(0, logliks1(i)-logliks0(i)+...
        log(normpdf(tf1Prop(2), 0, 1))- log(normpdf(tOut_i(2), 0, 1)));
    
    if (log(rand(1)) < A(i))
        tOut_i = tf1Prop;
        accept_i(2)  = 1;
        logliks0(i) = logliks1(i);
    end
    
    t1(i,:) = tOut_i;
    accept(i, :) = accept_i;
end

% reject if any scaling is negative
%     tCentered = [exp(log(t1(:,1))-mean(log(t1(:,1)))), t1(:,2)-mean(t1(:,2))];
tCentered  = [t1(:,1)./ mean(t1(:,1)), (t1(:,2) - mean(t1(:,2)))./mean(t1(:,1))];

tOut = transf0;
acceptOut = zeros(nsubj, 2);
wKY1 = wKY0;
if sum(tCentered(:,1)<0) == 0
    tOut = tCentered;
    coordTransfs = tCentered * [coord ones(length(coord),1)]';
    
    for i = 1:nsubj
        %         nnIDXs(i) = nns(nnDict,coord, coordTransfs(i,:)',incre, K);
        wKY1(:,:,i) = warpRBF_1d(coord, coordTransfs(i,:)', KY);
    end
    acceptOut = accept;
end

out.transf = tOut;
% out.nnIDXs = nnIDXs;
out.accept = acceptOut;
out.WKY = wKY1;
end