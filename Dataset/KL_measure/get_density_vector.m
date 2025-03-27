function [P Q]  = get_density_vector(P_temp, Q_temp)
% 
% input: two data (same columns) M1 x N or M2 x N
% N is the number of bands,  M is the number of samples
%
% output: two density propability for above data for KL distance
%

% compute the minmum and maximum value


N = size([P_temp; Q_temp], 1)/2;
bin = floor((1+log2(N)+0.5));

P_temp = P_temp(:)';
Q_temp = Q_temp(:)';
z_min = min([P_temp Q_temp]);
z_max = max([P_temp Q_temp]);


% get probability
P = histc(P_temp, z_min: (z_max-z_min)/(bin-1): z_max);
Q = histc(Q_temp, z_min: (z_max-z_min)/(bin-1): z_max);

P = P./sum(P) + eps;
Q = Q./sum(Q) + eps;


