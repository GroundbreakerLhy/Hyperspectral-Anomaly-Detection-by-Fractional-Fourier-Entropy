function BD_matrix = Multiclass_BD_v1(features, C)
%
% return the Bhattachar distance of a matrix: num_C x num_C
% feature: M x N
% N is the number of bands,  M is the number of samples
%

features = features./max(features(:));

num_class=length(C);
cum_C(1)=0;
cum_C(2:num_class+1)=cumsum(C);
BD_matrix = ones(num_class);
 
for i=1:length(cum_C)-1
    for j=1:length(cum_C)-1
        if i~=j
            P_temp = features((cum_C(i)+1): cum_C(i+1), :);
            Q_temp = features((cum_C(j)+1): cum_C(j+1), :);
        
            BD_matrix(i, j) = Bhattacharyya(P_temp', Q_temp');
        end
    end
end 