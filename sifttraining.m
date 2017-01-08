cd test
str='.jpg';
TRAINFEAT=zeros(328,1);
for i=35:37
    kk=num2str(i);
    x1=strcat(kk,str);
    x2=imread(x1);
    cd('D:\project2016_17\syed\maincode')
%% SIFT 
SI1=imresize(x2,[256 256]);
%% input image downsampling
Xdown = SI1(1:2:end,1:2:end,:);
% figure(4)
% imshow(Xdown);
% title('downsample image');
[m,n,o] = size(Xdown);
nblockcolumn = 2;
nblockrow = 2;
dcol = fix(n/nblockcolumn);
drow = fix(m/nblockrow);
xp=[];
for indexp = 1:nblockrow* nblockcolumn
 [r,c] = ind2sub([nblockrow,nblockcolumn],indexp );
subimage = Xdown((r-1)*drow+1:r*drow, (c-1)*dcol+1:c*dcol,:);
feature=siftrain(subimage,indexp);
cc=feature;
 xp=[xp feature];
end
xp=mean2(xp);
TRAINFEAT(i,1)=xp;
cd test
end
save TRAINFEAT