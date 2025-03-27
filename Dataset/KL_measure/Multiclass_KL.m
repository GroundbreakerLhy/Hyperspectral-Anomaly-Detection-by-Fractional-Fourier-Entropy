function KL_dist = Multiclass_KL(features, C)
%
% return the KL distance of data
% feature: M x N
% N is the number of bands,  M is the number of samples
%

features = features./max(features(:));
num_class=length(C);
cum_C(1)=0;
cum_C(2:num_class+1)=cumsum(C);
n=1;
 
for i=1:length(cum_C)-1
    for j=i+1:length(cum_C)-1
        P_temp = features((cum_C(i)+1): cum_C(i+1), :);
        Q_temp = features((cum_C(j)+1): cum_C(j+1), :);
        
        [P Q] = get_density_vector(P_temp, Q_temp);
        
        KL_temp(n) = KLDiv(Q, P);    % 'kl' or 'js'
        n=n+1;      
    end
end 

KL_dist =min(KL_temp);