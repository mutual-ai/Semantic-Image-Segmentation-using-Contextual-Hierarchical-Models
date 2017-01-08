close all;
clear all;
clc;
warning off;
%%------------------------set parameters---------------------%%

addpath('./others/');
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight 
alpha=0.99;% control the balance of two items in manifold ranking cost function
spnumber=200;% superpixel number
imgRoot='./test/';% test image path
saldir='./saliencymap/';% the output path of the saliency map
supdir='./superpixels/';% the superpixel label file path
mkdir(supdir);
mkdir(saldir);
imnames=dir([imgRoot '*' 'jpg']);

for ii=1:320
    disp(ii);
    imname=[imgRoot imnames(ii).name]; 
    [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame 
    [m,n,k] = size(input_im);

%%----------------------generate superpixels--------------------%%
    imname=[imname(1:end-4) '.bmp'];% the slic software support only the '.bmp' image
    comm=['SLICSuperpixelSegmentation' ' ' imname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];
    system(comm);    
    spname=[supdir imnames(ii).name(1:end-4)  '.dat'];
    superpixels=ReadDAT([m,n],spname); % superpixel label matrix
    spnum=max(superpixels(:));% the actual superpixel number

%%----------------------design the graph model--------------------------%%
% compute the feature (mean color in lab color space) 
% for each node (superpixels)
    input_vals=reshape(input_im, m*n, k);
    rgb_vals=zeros(spnum,1,3);
    inds=cell(spnum,1);
    for i=1:spnum
        inds{i}=find(superpixels==i);
        rgb_vals(i,1,:)=mean(input_vals(inds{i},:),1);
    end  
    lab_vals = colorspace('Lab<-', rgb_vals); 
    seg_vals=reshape(lab_vals,spnum,3);% feature for each superpixel
 
 % get edges
    adjloop=AdjcProcloop(superpixels,spnum);
    edges=[];
    for i=1:spnum
        indext=[];
        ind=find(adjloop(i,:)==1);
        for j=1:length(ind)
            indj=find(adjloop(ind(j),:)==1);
            indext=[indext,indj];
        end
        indext=[indext,ind];
        indext=indext((indext>i));
        indext=unique(indext);
        if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=i*ed(:,2);
            ed(:,1)=indext;
            edges=[edges;ed];
        end
    end

% compute affinity matrix
    weights = makeweights(edges,seg_vals,theta);
    W = adjacency(edges,weights,spnum);

% learn the optimal affinity matrix (eq. 3 in paper)
    dd = sum(W); D = sparse(1:spnum,1:spnum,dd); clear dd;
    optAff =(D-alpha*W)\eye(spnum); 
    mz=diag(ones(spnum,1));
    mz=~mz;
    optAff=optAff.*mz;
  
%%-----------------------------stage 1--------------------------%%
% compute the saliency value for each superpixel 
% with the top boundary as the query
    Yt=zeros(spnum,1);
    bst=unique(superpixels(1,1:n));
    Yt(bst)=1;
    bsalt=optAff*Yt;
    bsalt=(bsalt-min(bsalt(:)))/(max(bsalt(:))-min(bsalt(:)));
    bsalt=1-bsalt;

% down
    Yd=zeros(spnum,1);
    bsd=unique(superpixels(m,1:n));
    Yd(bsd)=1;
    bsald=optAff*Yd;
    bsald=(bsald-min(bsald(:)))/(max(bsald(:))-min(bsald(:)));
    bsald=1-bsald;
   
% right
    Yr=zeros(spnum,1);
    bsr=unique(superpixels(1:m,1));
    Yr(bsr)=1;
    bsalr=optAff*Yr;
    bsalr=(bsalr-min(bsalr(:)))/(max(bsalr(:))-min(bsalr(:)));
    bsalr=1-bsalr;
  
% left
    Yl=zeros(spnum,1);
    bsl=unique(superpixels(1:m,n));
    Yl(bsl)=1;
    bsall=optAff*Yl;
    bsall=(bsall-min(bsall(:)))/(max(bsall(:))-min(bsall(:)));
    bsall=1-bsall;   
   
% combine 
    bsalc=(bsalt.*bsald.*bsall.*bsalr);
    bsalc=(bsalc-min(bsalc(:)))/(max(bsalc(:))-min(bsalc(:)));
    
% assign the saliency value to each pixel     
     tmapstage1=zeros(m,n);
     for i=1:spnum
        tmapstage1(inds{i})=bsalc(i);
     end
     tmapstage1=(tmapstage1-min(tmapstage1(:)))/(max(tmapstage1(:))-min(tmapstage1(:)));
     
     mapstage1=zeros(w(1),w(2));
     mapstage1(w(3):w(4),w(5):w(6))=tmapstage1;
     mapstage1=uint8(mapstage1*255);  

     outname=[saldir imnames(ii).name(1:end-4) '_stage1' '.jpg'];
     imwrite(mapstage1,outname);

%%----------------------stage2-------------------------%%
% binary with an adaptive threhold (i.e. mean of the saliency map)
    th=mean(bsalc);
    bsalc(bsalc<th)=0;
    bsalc(bsalc>=th)=1;
    
% compute the saliency value for each superpixel
    fsal=optAff*bsalc;    
    
% assign the saliency value to each pixel
    tmapstage2=zeros(m,n);
    for i=1:spnum
        tmapstage2(inds{i})=fsal(i);    
    end
    tmapstage2=(tmapstage2-min(tmapstage2(:)))/(max(tmapstage2(:))-min(tmapstage2(:)));

    mapstage2=zeros(w(1),w(2));
    mapstage2(w(3):w(4),w(5):w(6))=tmapstage2;
    mapstage2=uint8(mapstage2*255);
    outname=[saldir imnames(ii).name(1:end-4) '_stage2' '.jpg'];   
    imwrite(mapstage2,outname);
    
    
end
%% 
str='.jpg';
for ih=1:320
    cd segmented_images

GG1=num2str(ih);
    im_se=strcat(GG1,str);
     im_se1=imread( im_se);
segmented_image=im2bw( im_se1);
[aa bb]=size(segmented_image);
cd('D:\project2016_17\syed\maincode');
cd groundtruth
GG=num2str(ih);
    xs1=strcat(GG,str);
    GTM=imread(xs1);
    GTM=imresize(GTM,[aa bb]);
    GTM=im2bw(GTM);
    Final_seg=imfuse(GTM,segmented_image);
    Final_seg=im2bw(Final_seg);
    cd('D:\project2016_17\syed\maincode');
    cd outputimages
    outname=strcat(num2str(ih),str); 
    imwrite( Final_seg,outname);
    cd('D:\project2016_17\syed\maincode');
end
    cd('D:\project2016_17\syed\maincode');
%% SIFT 
% rng default
format long
cd test
% x1=uigetfile('*.jpg');
x1=imread('1.jpg');
SI1=imresize(x1,[256 256]);
 cd('D:\project2016_17\syed\maincode');
%% input image downsampling
Xdown = SI1(1:2:end,1:2:end,:);
figure(4)
imshow(Xdown);
title('downsample image');
[m,n,o] = size(Xdown);
nblockcolumn = 2;
nblockrow = 2;
dcol = fix(n/nblockcolumn);
drow = fix(m/nblockrow);
xp=[];
for indexp = 1:nblockrow* nblockcolumn
 [r,c] = ind2sub([nblockrow,nblockcolumn],indexp );
subimage = Xdown((r-1)*drow+1:r*drow, (c-1)*dcol+1:c*dcol,:);
feature=sif(subimage,indexp);
cc=feature;
 xp=[xp feature];
end
TNR=0.95;
xp=mean2(xp);
x2=double(x1);
VA=mean2(var(x2));
SK=mean2(skewness(x2));
disp('SIFT feature');
disp(xp);
disp('Variance');
disp(VA);
disp('Skewness');
disp(SK);
inpf=[xp VA SK];
numofimgs=300;
datasetnn=xlsread('DATA.xlsx');
datasetnn=datasetnn(1:60,:);
inpf1=datasetnn(55,:);
metric=1;
[P, R, cmat] = svm(numofimgs,datasetnn,inpf1,metric);
 P(isnan(P))=1;
 R(isnan(R))=1;
    Precision=mean(P)
Recall=mean(R)
F_value=(2*Precision*Recall)/(Precision+Recall)
G_mean=sqrt(Recall*TNR)