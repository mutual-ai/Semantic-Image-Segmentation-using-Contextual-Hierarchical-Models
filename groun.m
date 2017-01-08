%% groundtruth
y1=imread('g1.jpg');
y1=imresize(y1,[256 256]);
figure
imshow(y1);
[gm,gn,go] = size(y1);
title('groundtruth image');
yp=[];
gnblockcolumn = 2;
gnblockrow = 2;
gdcol = fix(gn/gnblockcolumn);
gdrow = fix(gm/gnblockrow);

for inde = 1:gnblockrow*gnblockcolumn
    
[gr,gc] = ind2sub([gnblockrow,gnblockcolumn],inde );

groundsubimage = y1((gr-1)*gdrow+1:gr*gdrow, (gc-1)*gdcol+1:gc*gdcol,:);
subplot(gnblockrow,gnblockcolumn,inde);
imshow(groundsubimage)
feature=gsif(groundsubimage,inde);
cc=feature;
 yp=[yp feature];

end
%% groudtruth classification


inputs1=yp;
   targets1=exp(inputs1);
   hiddenLayerSize = 10;
net1 = patternnet(hiddenLayerSize);
[net1,tr1] = train(net1,inputs1,targets1);
outputs1 = net1(inputs1);
save outputs1
%% max pool for lower level classifier
max_pool=[];
kk=max(outputs1);
max_pool=[max_pool kk];