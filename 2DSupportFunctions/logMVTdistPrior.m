function out = logMVTdistPrior(Tvec, priors, ta0, tb0, S)

Tmat = Vec2Mat(Tvec, false);
PriorMat = Vec2Mat(priors, false);
% centered = Tmat*inv(PriorMat) - eye(3);
centered = Tmat-PriorMat;
out = log(my_mvtpdf(centered(:)', ta0/tb0*S, 2*ta0));
end