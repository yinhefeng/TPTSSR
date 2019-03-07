function M = AdaptiveSupervised(Xfea, Xgnd, MArray)
% Adaptive Supervised self-tuning scheme for the TPTSSR parameter M.
%
% 
%         Input:
%           Xfea            - Train Matrix (each column represent a sample).
%           Xgnd            - Label vector containing the labels of Xfea matrix.
%           MArray          - Array of possible values for M
% 
% 
% 
%         Output:
%           M               - Estimated number of samples that should be moved to the second phase of TPTSSR. 

N = size(Xfea, 2);

for i=1:N
    Xf=Xfea; Xg=Xgnd;
    yfea=Xf(:,i); ygnd=Xg(:,i);
    Xf(:,i)=[]; Xg(:,i)=[];
    cr=zeros(length(MArray),1);
    for m = 1:length(MArray)
        [Deviation, SelectedClass] = TPTSSRdeviation(Xf, Xg, yfea, MArray(m));
        cr(m)=Deviation;
        if SelectedClass ~= ygnd
            cr(m)=inf;
        end
    end
    [~,index]=min(cr);
    M(i)=MArray(index);
    if min(cr)==inf
        M(i)=0;
    end
end

end




function [Deviation, SelectedClass] = TPTSSRdeviation(Xfea, Xgnd,y,M)

for i=1:size(Xfea, 2)
    Xfea(:,i)=Xfea(:,i)/norm(Xfea(:,i));
end
y=y/norm(y);

[Z,CZ] = FirstPhase(Xfea, Xgnd,y,M);
[Deviation, SelectedClass] = SecondPhase(Z,CZ,y);
end

function [Z,CZ] = FirstPhase(Xfea, Xgnd, y, M)
mu=0.01;
n = size(Xfea,2);
e=zeros(1,n);

A=(Xfea'*Xfea+mu*eye(n))\(Xfea'*y);

for i=1:n
    e(i)=norm(y-A(i)*Xfea(:,i))^2;
end

[~, index]=sort(e,'ascend');

Z=Xfea(:,index(1:M));
CZ=Xgnd(index(1:M));

end

function [Deviation, SelectedClass] = SecondPhase(Z,CZ,y)

mu=0.01;
M = size(Z,2);

B=(Z'*Z+mu*eye(M))\(Z'*y);

UnrepeatedCz = unique(CZ);
nbC=length(UnrepeatedCz);

D=zeros(1,nbC);
g=zeros(size(Z,1),nbC);
for j=1:nbC
    for k=1:M
        if CZ(k)==UnrepeatedCz(j)
            g(:,j)=g(:,j)+B(k)*Z(:,k);
        end
    end
    D(j)=norm(y-g(:,j))^2;
end

[D,index]=sort(D);

SelectedClass = UnrepeatedCz(index(1));
Deviation = D(1);
end
