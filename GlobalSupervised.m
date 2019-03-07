function M = GlobalSupervised(Xfea, Xgnd, MArray)
% Global Supervised self-tuning scheme for the TPTSSR parameter M.
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
f = zeros(length(MArray),1);

for m = 1:length(MArray)
    succeed=0;
    for i = 1:N
        Xf=Xfea; Xg=Xgnd;
        y.fea=Xf(:,i); y.gnd=Xg(:,i);
        Xf(:,i)=[]; Xg(:,i)=[];
        succeed = succeed + (TPTSSR(Xf, Xg,y.fea,MArray(m)) == y.gnd);
    end
    f(m)=succeed;
end
[~,index]=max(f);
M=MArray(index);


end
