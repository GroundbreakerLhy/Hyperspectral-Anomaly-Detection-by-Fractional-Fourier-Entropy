function KL_matrix = Multiclass_KL_v1(features, C)
%
% return the KL distance of a matrix: num_C x num_C
% feature: M x N
% N is the number of bands,  M is the number of samples
%

num_class=length(C);
cum_C(1)=0;
cum_C(2:num_class+1)=cumsum(C);
KL_matrix = ones(num_class);
 
for i=1:length(cum_C)-1
    for j=1:length(cum_C)-1
        if i~=j
            P_temp = features((cum_C(i)+1): cum_C(i+1), :);
            Q_temp = features((cum_C(j)+1): cum_C(j+1), :);
        
            [P Q] = get_density_vector(P_temp, Q_temp);
        
            KL_matrix(i, j) = KLDiv(Q, P);    % 'kl' or 'js'
        end
    end
end 