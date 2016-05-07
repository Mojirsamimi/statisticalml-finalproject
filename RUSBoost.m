% clear;clc
%% Load and Preprocess Data
disp('load/preprocess');
load featureMap.mat;
test=csvread('../test.csv',1,1);
train=csvread('../train.csv',1,1);

test_data_file=csvread('../test.csv',1,1);
train_data_file=csvread('../train.csv',1,1);

TrainData=train_data_file(:,1:end-1);
TrainLabels=train_data_file(:,end);
TestData=test_data_file;
% add number of zeros in each row as a new feature n0
n0ind=TrainData==0;
n0=sum(n0ind,2);
TrainData=[TrainData n0];

n0ind=TestData==0;
n0=sum(n0ind,2);
TestData=[TestData n0];
featureMap('n0')=370;
clear n0ind n0;
% we split up var38 into two variables
% var38mc == 1 when var38 has the most common value and 0 otherwise
% logvar38 is log transformed feature when var38mc is 0, zero otherwise
var38=TrainData(:,featureMap('var38'));
var38mc=zeros(size(var38));
logvar38=zeros(size(var38));
commonValInd=abs(var38-117310.979016)<=1;
var38mc(commonValInd)=1;
logvar38(~commonValInd)=log(var38(~commonValInd));
TrainData=[TrainData logvar38 var38mc];
% Test Data
var38=TestData(:,featureMap('var38'));
var38mc=zeros(size(var38));
logvar38=zeros(size(var38));
commonValInd=abs(var38-117310.979016)<=1;
var38mc(commonValInd)=1;
logvar38(~commonValInd)=log(var38(~commonValInd));
TestData=[TestData logvar38 var38mc];

featureMap('logvar38')=371;
featureMap('var38mc')=372;

trainVar3=TrainData(:,featureMap('var3'));
indx=trainVar3==-999999;
TrainData(indx,featureMap('var3'))=2;

trainVar3=TestData(:,featureMap('var3'));
indx=trainVar3==-999999;
TestData(indx,featureMap('var3'))=2;

clear commonValInd var38mc logvar38 trainVar3 indx;
%%
disp('feature');
idxvar(1,end)=371;
idxvar=[idxvar 372];
TrainData=TrainData(:,idxvar);
TestData=TestData(:,idxvar);
%%
disp('RUS');
t = templateTree('MinLeafSize',5);
tic
rusBoost = fitensemble(TrainData,TrainLabels,'RUSBoost',1000,t,...
    'LearnRate',0.1,'nprint',100,'RatioToSmallest',[10 1],'Cost',[0 1;10 0],'CrossVal','on','KFold',5);
toc

[RUSYfit,RUSscores]=rusBoost.kfoldPredict();
%%
[X,Y,T,AUC] = perfcurve(TrainLabels,RUSscores(:,2),1);
AUC