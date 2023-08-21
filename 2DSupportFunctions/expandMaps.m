function LLmap = expandMaps(Latent, L, nsubj)
nx = size(Latent, 1);
ny = size(Latent, 2);  
LLmap = zeros(nx+2*L, ny+2*L, nsubj);
LLmap((L+1):(nx+L), (L+1):(ny+L), :) = Latent;
LLmap = squeeze(LLmap);
end