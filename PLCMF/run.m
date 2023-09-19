clear;
addpath('.\tool');
rand('seed',900);
%load('HandWritten.mat');
%for i=1:size(X,2)
%    X{i}=mapminmax(X{i},-1,1);
%%     X{i}=NormalizeData(X{i});
%end

load(['D:/cyy/dataset/MVC_data/','Caltech-5V','.mat'],'X1','X2','X3','X4','X5','Y') % 出来的是n * d
gt = double(transpose(Y) + 1);
%X = {double(transpose(X1)),double(transpose(X2))};
%X = {double(transpose(X1)),double(transpose(X2)),double(transpose(X5))}
%X = {double(transpose(X1)),double(transpose(X2)),double(transpose(X5)),double(transpose(X3))}
%X = {double(transpose(X1)),double(transpose(X2)),double(transpose(X5)),double(transpose(X3)),double(transpose(X4))}

load(['D:/cyy/dataset/MVC_data/','BDGP','.mat'],'X1','X2','Y') % 出来的是n * d
gt = double(transpose(Y) + 1);
X = {double(transpose(X1)),double(transpose(X2))};

for i=1:size(X,2)
    X{i}=mapminmax(X{i},-1,1); % 需要的是d * n
%     X{i}=NormalizeData(X{i});
end

option.numClust = size(unique(gt),1);
option.threshold=1e-1;              
option.K=100;
option.delta=1e-1; 
option.beta=1e-2;  
option.lambda=1e-4;            
option.r=5; 
option.max_iter = 20; 
option.Vnum = size(X,2);
option.N = size(X{1},2);                            
option.alpha = ones(option.Vnum,1) / (option.Vnum); 
[result]=PLCMF(X,gt,option);
fprintf('\nAll view results: ACC = %.4f, NMI = %.4f, Purity = %.4f, F-score = %.4f, Precision = %.4f and , Recall = %.4f\n',result(1),result(2),result(3),result(4),result(5),result(6));

 

