function M = AdaptiveSemiSupervised(Xfea, Xgnd, labeled_mask, MArray)
% Adaptive Semi-supervised self-tuning scheme for the TPTSSR parameter M.
%
% 
%         Input:
%           Xfea            - Train Matrix (each column represent a sample).
%           Xgnd            - Label vector containing the labels of Xfea matrix.
%           labeled_mask    - A binary vector clarifying the labeled and unlabeled data,
%                             0 means the correspond data is unlabeled and 1 means
%                             the correspond data is labeled. Size of the vector is
%                             1xP. The same as labels vector.
%           MArray          - Array of possible values for M
% 
% 
% 
%         Output:
%           M               - Estimated number of samples that should be moved to the second phase of TPTSSR. 

N = size(Xfea, 2);

data=Xfea;
[ ~ , nSmp] = size(data);
aa = sum(data .* data);
ab = data' * data;
M_distance =      repmat(aa',1,nSmp) + repmat(aa, nSmp,1) - 2*ab';
M_distance (abs(M_distance )<1e-10)=0;
dmax=max(max(M_distance));

for i = 1:N
    CS=i;
    M_dist = M_distance(:,i);    
    for j=1:N
        M_dist(CS,1)=inf;
        [~,index]=min(M_dist);
        D(i,j)=min(M_distance(index,CS));
        CS=[CS,index];
    end
end

D(:,end)=[];
for i = 1:N
    for j=MArray
        if D(i,j)>mean(D(i,:))
            M(i,1)=j;
            break;
        end
        M(i,1)=j;
    end
end

    
nbClass = max(Xgnd);

lbfea = Xfea(:,labeled_mask==1);
lbgnd = Xgnd(:,labeled_mask==1);

l = sum(lbgnd==1);
u=sum(Xgnd==1)-l;

M2 = AdaptiveSupervised(lbfea, lbgnd, MArray);

M=[M;M2'];
% M2 = reshape(M2,l,nbClass);
% M2 = [M2;zeros(u,nbClass)];
% M(:,2) = M2(:);

end



function M = AdaptiveSupervised(Xfea, Xgnd,MTable)
N = size(Xfea, 2);

for i=1:N
    Xf=Xfea; Xg=Xgnd;
	y.fea=Xf(:,i); y.gnd=Xg(:,i);
    Xf(:,i)=[]; Xg(:,i)=[];
    cr=zeros(length(MTable),1);
    for m = 1:length(MTable)
        [Deviation, SelectedClass] = TPTSSRdeviation(Xf, Xg, y.fea,MTable(m));
        cr(m)=Deviation;
        if SelectedClass ~= y.gnd
            cr(m)=inf;
        end
    end
    [~,index]=min(cr);
    M(i)=MTable(index)+1;
    if min(cr)==inf
        M(i)=0;
    end
    disp([num2str(i),'/', num2str(N)]);
end

M2=M; M2(M2==0)=[];
M(M==0)=floor(mean(M2));

end




function [Deviation, SelectedClass] = TPTSSRdeviation(Xfea, Xgnd, y, M)

for i=1:size(Xfea, 2)
    Xfea(:,i)=Xfea(:,i)/norm(Xfea(:,i));
end
y=y/norm(y);

[Z,CZ] = FirstPhase(Xfea, Xgnd, y, M);
[Deviation, SelectedClass] = SecondPhase(Z,CZ,y);
end

function [Z,CZ] = FirstPhase(Xfea, Xgnd,y,M)
mu=0.01;
n = size(Xfea,2);
e=zeros(1,n);

A=(Xfea'*Xfea+mu*eye(n))\(Xfea'*y);

for i=1:n
    e(i)=norm(y-A(i)*Xfea(:,i))^2;
end

[~,index]=sort(e,'ascend');

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

