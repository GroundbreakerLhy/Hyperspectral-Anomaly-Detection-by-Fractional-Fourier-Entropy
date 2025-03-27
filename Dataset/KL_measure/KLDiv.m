function KL = KLDiv(P1, P2, varargin)
%
%  KL = KLDiv(P1, P2) Kullback-Leibler divergence of two discrete probability
%  distributions:
%  P1 : 1 x n         P2 : 1 x n
%
%  The Kullback-Leibler divergence is given by:
%       KL(P1(x),P2(x)) = sum[P1(x).log(P1(x)/P2(x))]
% dist = n x 1

if size(P1,2)~=size(P2,2)
    error('the number of columns in P1 and P2 should be the same');
end

% Check probabilities sum to 1:
if (abs(sum(P1) - 1) > .00001) || (abs(sum(P2) - 1) > .00001),
    error('Probablities don''t sum to 1.')
end

% if ~isempty(varargin),
%     switch varargin{1},
%         case 'js',
%             logQvect = log2((P2+P1)/2);
%             KL = .5 * (sum(pVect1.*(log2(P1)-logQvect)) + ...
%                 sum(P2.*(log2(P2)-logQvect)));
% 
%         case 'sym',
%             KL1 = sum(P1 .* (log2(P1)-log2(P2)));
%             KL2 = sum(P2 .* (log2(P2)-log2(P1)));
%             KL = (KL1+KL2)/2;
%             
%         otherwise
%             error(['Last argument' ' "' varargin{1} '" ' 'not recognized.'])
%     end
% else
%     KL = sum(P1 .* (log2(P1)-log2(P2)));
% end

N = length(P1);
KL = zeros(1, N);
for i = 1: N
    if (P1(i) > 0) && (P2(i) > 0)
        KL(i) = P1(i) * (log2(P1(i) - log2(P2(i))));
    end
end

KL = sum(KL);
