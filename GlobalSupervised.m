function M = GlobalSupervised(Xfea, Xgnd, MTable)
N = size(Xfea, 2);
f = zeros(length(MTable),1);

for m = 1:length(MTable)
    succeed=0;
    for i = 1:N
        Xf=Xfea; Xg=Xgnd;
        y.fea=Xf(:,i); y.gnd=Xg(:,i);
        Xf(:,i)=[]; Xg(:,i)=[];
        succeed = succeed + (TPTSSR(Xf, Xg,y.fea,MTable(m)) == y.gnd);
    end
    f(m)=succeed;
end
[~,index]=max(f);
M=MTable(index);


end