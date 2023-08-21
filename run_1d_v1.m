% simulate 1d curves
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\");
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\MCMC\supportFunctions\");
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\MCMC\2DSupportFunctions\");
addpath("D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\code\Matlab\MCMC\RBF");
resultsDir = '/Users/guoqingwang/Dropbox/projects/SpatialProjectGIT/SpatialAnalysis/GroupwiseReg/results/';

%%

shifts = [-2, 0, 2];
scalings = [.8,1,1.2];
coords = (-4:.1:4)';
% dat = load('D:\Dropbox\projects\SpatialProjectGIT\SpatialAnalysis\GroupwiseReg\data\1d_cosine_data.mat').dat';
dat = load('../data/1d_cosine_data.mat').dat';
eps=0.1;
figure; plot(coords, dat);legend(["subj 1", "subj 2", "subj 3"]);

figure('Position', [1000,1000, 1e3, 2.5e2]);
for i=1:3
subplot(1,3,i);
plot(coords, dat(:,i), 'LineWidth',1);
ylim([-1.5, 2.5]);
title(['Curve ',num2str(i)]);
end

%% initialization and input
init.b = [1,1,1];
init.shift = shifts;
init.scaling = scalings;
init.rot = [0, 0, 0];

init.X = dat(:,1);
init.sigma = [1e-3, 1e-3, 1e-3];
init.alpha = .5;
init.rho = 0.5;
% dataInput
dataIn.Y = dat;
dataIn.coord = coords;
dataIn.nsubj = int32(3);
dataIn.K = 10;
dataIn.nnX = nnsearch(dataIn.coord , dataIn.K);
dataIn.nnDict = nnDictionary(dataIn.coord, 10, dataIn.K);
%% conventional method
hyperPars.shape = .05;%.0001
hyperPars.KN = 1;
niter = 1e4/2;
rng(1);
out1d = MCMC1d_rbf(init, dataIn, niter, hyperPars);

%% trace plot

nburnin =3e3;
range = nburnin:niter;

means = mean(out1d.X(range, :), 1);
stds = std(out1d.X(range, :),0,1);
xconf = [dataIn.coord' dataIn.coord(end:-1:1)'];
yconf = [means+1.96*stds means(end:-1:1)-1.96*stds];

figure;
p = fill(xconf,yconf,'red');
p.FaceColor = [1 0.8 0.8];      
p.EdgeColor = 'none';  
hold on
plot(dataIn.coord, means,'r', 'LineWidth', 1);
plot(results1.coordinates, results1.Truth,'k', 'LineWidth', 1);
hold off
ylim([-0.5,1.5])
legend(["Fitted CI", "Fitted Mean", "True template"]);

%% proposed method

rng("default");
niter = 1e4;
out1d_nngp = MCMC1d(init, dataIn, niter);

%%  plot
nburnin =6e3;

means = mean(out1d_nngp.X(nburnin:end, :), 1);
stds = std(out1d_nngp.X(nburnin:end, :),0,1);
xconf = [dataIn.coord' dataIn.coord(end:-1:1)'];
yconf = [means+1.96*stds means(end:-1:1)-1.96*stds];

figure;
p = fill(xconf,yconf,'red');
p.FaceColor = [1 0.8 0.8];      
p.EdgeColor = 'none';  
hold on
plot(dataIn.coord, means,'r', 'LineWidth', 1);
plot(dataIn.coord, meanCurve,'k', 'LineWidth', 1);
hold off
ylim([-1,2])
legend(["Fitted CI", "Fitted Mean", "True template"]);

