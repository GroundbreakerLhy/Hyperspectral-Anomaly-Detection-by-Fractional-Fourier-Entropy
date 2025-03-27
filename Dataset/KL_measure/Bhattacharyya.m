function y = Bhattacharyya(Y1,Y2)
%-------------------------------------------------
% Y1: num_dim x num_sam
% y = Bhattacharyys(Y1,Y2);
%       M1 : is the 1xd mean vector of class 1.
%       M2 : is the 1xd mean vector of class 2.
%     Cov1 : is the covariance matrix for class1.
%     Cov2 : is the covariance matrix for class2.
% output y : is the minimum Bhattacharyya distance
%-------------------------------------------------

M1 = mean(Y1,2);
M2 = mean(Y2,2);
Cov1 = cov(Y1');
Cov2 = cov(Y2');
temp = pinv((Cov1 + Cov2)/2);
y = 1 / 8 * (M2 - M1)' * temp * (M2 - M1) +...
    1 / 2 * log(norm((Cov1+Cov2)/2)/sqrt(norm(Cov1)*norm(Cov2)));