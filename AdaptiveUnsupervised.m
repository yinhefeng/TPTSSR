function M = AdaptiveUnsupervised(Xfea,MTable)

N = size(Xfea, 2);

data=Xfea;
[ ~ , nSmp] = size(data);
aa = sum(data .* data);
ab = data' * data;
M_distance =      repmat(aa',1,nSmp) + repmat(aa, nSmp,1) - 2*ab';
M_distance (abs(M_distance )<1e-10)=0;

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
    for j=1:max(MTable)
        if D(i,j)>mean(D(i,1:max(MTable)))
            M(i)=j;
            break;
        end
    end
end


end
