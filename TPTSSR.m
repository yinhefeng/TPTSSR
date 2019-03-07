function SelectedClass = TPTSSR(Xfea, Xgnd,y,M)
%%
%  This function represents the algorithm of the TPTSSR method
%   Inputs    TrainMatrix : is a structure of training data, contains
%                       fea : trining Matrix, each column is an observation
%                       gnd : ground truth or label of coresponding column
%             y : test column vector
%             M : number of nearest neighbors should be selected

%     Output
%             SelectedClass : the selected class for the test vector y
%

%   Sample
%            SelectedClass = TPTSSR(TrainMatrix,y,M);
%    executing the TPTSSR method and saving output value
%

for i=1:size(Xfea, 2)
    Xfea(:,i)=Xfea(:,i)/norm(Xfea(:,i));
end
y=y/norm(y);

%% Executing of the first phase
[Z,CZ] = FirstPhase(Xfea, Xgnd,y,M);
%% Executing of the second phase
SelectedClass = SecondPhase(Z,CZ,y);

end

function [Z,CZ] = FirstPhase(Xfea, Xgnd,y,M)
%% Definition and preparation of variables
mu=0.01;
% TrainMatrix.fea = zscore(TrainMatrix.fea);
n = size(Xfea,2);
% m = size(TrainMatrix.fea,1);
e=zeros(1,n);

%% Searching the M nearest neighbors
A=(Xfea'*Xfea+mu*eye(n))\(Xfea'*y); % size(A)=(n,1)

% e1=norm(repmat(y',n,1)-repmat(A,1,m).*TrainMatrix.fea');
for i=1:n
    e(i)=norm(y-A(i)*Xfea(:,i))^2; % size(e)=(n)
end

[~,index]=sort(e,'ascend');

%% Saving the M nearest neighbors in outputs
Z=Xfea(:,index(1:M));
CZ=Xgnd(index(1:M));

end

function SelectedClass = SecondPhase(Z,CZ,y)
%% Definition and preparation of variables

mu=0.01;
% Z = zscore(Z);
M = size(Z,2);

%% searching the class having the greater contribution with the test sample
B=(Z'*Z+mu*eye(M))\(Z'*y); % size(B)=(M,1)

UnrepeatedCz = unique(CZ);
nbC=length(UnrepeatedCz);% nbC = number of class

D=zeros(1,nbC);
g=zeros(size(Z,1),nbC);
for j=1:nbC
    for k=1:M
        if CZ(k)==UnrepeatedCz(j) % which means Z(k) belongs to the Class j.
            g(:,j)=g(:,j)+B(k)*Z(:,k);
        end
    end
    D(j)=norm(y-g(:,j))^2;
end

[~,index]=sort(D);

%% Saving the selected class as output
SelectedClass = UnrepeatedCz(index(1));
end