function q = DFrFT2D(img,a)
[m,n]=size(img);

if ~isa(img,'double')
    img=double(img);
end
P=zeros(m,n);
for i=1:m
    t=Disfrft(img(i,:),a);
    P(i,:)=t;
end
Q=zeros(m,n);
for i=1:n
    t=Disfrft(P(:,i),a);
    Q(:,i)=t;
end

q=Q;
end

