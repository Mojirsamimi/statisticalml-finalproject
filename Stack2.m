%% Feuture map
clear;clc
load FeaturesIndx.mat
load featureMap.mat;
%% Loading data
disp('load');
test_data_file=csvread('../test.csv',1,1);
train_data_file=csvread('../train.csv',1,1);
trainA_file=csvread('../trainA.csv',1,1);
trainB_file=csvread('../trainB.csv',1,1);

TrainData=train_data_file(:,1:end-1);
TrainLabels=train_data_file(:,end);
TestData=test_data_file;
TrainDataA=trainA_file(:,1:end-1);
TrainLabelsA=trainA_file(:,end);
TrainDataB=trainB_file(:,1:end-1);
TrainLabelsB=trainB_file(:,end);
% add number of zeros in each row as a new feature n0
n0ind=TrainData==0;
n0=sum(n0ind,2);
TrainData=[TrainData n0];

n0ind=TestData==0;
n0=sum(n0ind,2);
TestData=[TestData n0];

n0ind=TrainDataA==0;
n0=sum(n0ind,2);
TrainDataA=[TrainDataA n0];

n0ind=TrainDataB==0;
n0=sum(n0ind,2);
TrainDataB=[TrainDataB n0];
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

var38=TrainDataA(:,featureMap('var38'));
var38mc=zeros(size(var38));
logvar38=zeros(size(var38));
commonValInd=abs(var38-117310.979016)<=1;
var38mc(commonValInd)=1;
logvar38(~commonValInd)=log(var38(~commonValInd));
TrainDataA=[TrainDataA logvar38 var38mc];

var38=TrainDataB(:,featureMap('var38'));
var38mc=zeros(size(var38));
logvar38=zeros(size(var38));
commonValInd=abs(var38-117310.979016)<=1;
var38mc(commonValInd)=1;
logvar38(~commonValInd)=log(var38(~commonValInd));
TrainDataB=[TrainDataB logvar38 var38mc];

featureMap('logvar38')=371;
featureMap('var38mc')=372;
clear commonValInd var38mc logvar38;


%% Preprocess Features
% var3, value -999999 replaced with most common value 2
% TrainData=train_data_file;
% TestData=test_data_file;
% TrainLabels=train_data_file(:,end);
trainVar3=TrainData(:,featureMap('var3'));
indx=trainVar3==-999999;
TrainData(indx,featureMap('var3'))=2;
trainVar3=TrainDataA(:,featureMap('var3'));
indx=trainVar3==-999999;
TrainDataA(indx,featureMap('var3'))=2;
trainVar3=TrainDataB(:,featureMap('var3'));
indx=trainVar3==-999999;
TrainDataB(indx,featureMap('var3'))=2;
trainVar3=TestData(:,featureMap('var3'));
indx=trainVar3==-999999;
TestData(indx,featureMap('var3'))=2;
clear trainVar3 indx;

%% Most important features kaggle forum

% featureIndx=cell2mat(values(featureMap,{'logvar38','var38mc','var15','ind_var5', 'ind_var8_0', 'ind_var30', 'num_var5', 'num_var30', 'num_var42', 'var36', 'num_meses_var5_ult3'}));
%                                                            
% TrainData=TrainData(:,featureIndx);
% TrainDataA=TrainDataA(:,featureIndx);
% TrainDataB=TrainDataB(:,featureIndx);
% TestData=TestData(:,featureIndx);
%% Most important features viet

% featureIndx=cell2mat(values(featureMap,{'logvar38','var38mc','var15', 'imp_op_var41_efect_ult1', 'ind_var12', 'ind_var13', 'ind_var24', 'ind_var30_0', 'ind_var30', 'saldo_var30', 'num_var22_ult3', 'num_meses_var5_ult3'}));
%                                                             
% TrainData=TrainData(:,featureIndx);
% TrainDataA=TrainDataA(:,featureIndx);
% TrainDataB=TrainDataB(:,featureIndx);
% TestData=TestData(:,featureIndx);

%% Most important features me
disp('feature');
idxvar(1,end)=371;
idxvar=[idxvar 372];
TrainData=TrainData(:,idxvar);
TrainDataA=TrainDataA(:,idxvar);
TrainDataB=TrainDataB(:,idxvar);
TestData=TestData(:,idxvar);
%% Training RUSBoost
disp('RUS');
t = templateTree('MinLeafSize',5);
tic
rusTreeA = fitensemble(TrainDataA,TrainLabelsA,'RUSBoost',1000,t,...
    'LearnRate',0.1,'nprint',100,'RatioToSmallest',[10 1],'Cost',[0 1;10 0]); %5 0.9550,8 0.9561,9 0.9566,10 0.9571,11 0.9567,15 0.9564,20 0.9556,24 0.9508
toc


[RUSYfitB,RUSscoresB]=rusTreeA.predict(TrainDataB);
SumOfClasses=RUSscoresB(1,1)+RUSscoresB(1,2);
RUSscoresB=RUSscoresB./SumOfClasses;

tic
rusTreeB = fitensemble(TrainDataB,TrainLabelsB,'RUSBoost',1000,t,...
    'LearnRate',0.1,'nprint',100,'RatioToSmallest',[10 1],'Cost',[0 1;10 0]);
toc

[RUSYfitA,RUSscoresA]=rusTreeB.predict(TrainDataA);
SumOfClasses=RUSscoresA(1,1)+RUSscoresA(1,2);
RUSscoresA=RUSscoresA./SumOfClasses;

t = templateTree('MinLeafSize',5);
tic
rusTree = fitensemble(TrainData,TrainLabels,'RUSBoost',1000,t,...
    'LearnRate',0.1,'nprint',100,'RatioToSmallest',[10 1],'Cost',[0 1;10 0]);
toc

[RUSYfit,RUSscores]=rusTree.predict(TestData);
SumOfClasses=RUSscores(1,1)+RUSscores(1,2);
RUSscores=RUSscores./SumOfClasses;

dlmwrite('Stack3/RUSBoostA.csv', RUSscoresA, 'delimiter', ',', 'precision', 10); 
dlmwrite('Stack3/RUSBoostB.csv', RUSscoresB, 'delimiter', ',', 'precision', 10);
dlmwrite('Stack3/RUSBoostTest.csv', RUSscores, 'delimiter', ',', 'precision', 10);


clear RUSscoresA RUSscoresB RUSscores rusTree rusTreeA rusTreeB RUSYfit RUSYfitA RUSYfitB
%% Training Bag
disp('Bag');
bagA = fitensemble(TrainDataA,TrainLabelsA,'Bag',200,'Tree','Type','Classification','Cost',[0 1;10 0]);

[BagYB,BagScoreB]=bagA.predict(TrainDataB);

bagB = fitensemble(TrainDataB,TrainLabelsB,'Bag',200,'Tree','Type','Classification','Cost',[0 1;10 0]);

[BagYA,BagScoreA]=bagB.predict(TrainDataA);

bag = fitensemble(TrainData,TrainLabels,'Bag',200,'Tree','Type','Classification','Cost',[0 1;10 0]);

[BagY,BagScore]=bag.predict(TestData);

dlmwrite('Stack3/BagA.csv', BagScoreA(:,2), 'delimiter', ',', 'precision', 10); 
dlmwrite('Stack3/BagB.csv', BagScoreB(:,2), 'delimiter', ',', 'precision', 10);
dlmwrite('Stack3/BagTest.csv', BagScore(:,2), 'delimiter', ',', 'precision', 10);

clear BagScoreA BagScoreB BagScore BagY BagYA BagYB bagA bagB bag
%% Training SVM
disp('SVM');
SVMA = fitcsvm(TrainDataA,TrainLabelsA,'KernelFunction','rbf','Standardize',true,'ClassNames',[0,1],'Cost',[0 1;10 0]);
% mdlSVM = fitPosterior(SVMA);
% [~,score_svm] = resubPredict(mdlSVM);
[SVMyB,SVMScoreB]=SVMA.predict(TrainDataB);
% [SVMyB2,SVMScoreB2]=mdlSVM.predict(TrainDataB);

SVMB = fitcsvm(TrainDataB,TrainLabelsB,'KernelFunction','rbf','Standardize',true,'ClassNames',[0,1],'Cost',[0 1;10 0]);

[SVMyA,SVMScoreA]=SVMB.predict(TrainDataA);

SVM = fitcsvm(TrainData,TrainLabels,'KernelFunction','rbf','Standardize',true,'ClassNames',[0,1],'Cost',[0 1;10 0]);

[SVMy,SVMScore]=SVM.predict(TestData);

dlmwrite('Stack3/SvmA.csv', SVMScoreA(:,2), 'delimiter', ',', 'precision', 10); 
dlmwrite('Stack3/SvmB.csv', SVMScoreB(:,2), 'delimiter', ',', 'precision', 10);
dlmwrite('Stack3/SvmTest.csv', SVMScore(:,2), 'delimiter', ',', 'precision', 10);

 clear SVMScore SVMScoreB SVMScoreA SVMy SVMyA SVMyB SVM SVMA SVMB
%% Training AdaBoost
disp('AdaBoost');
AdaBoostM1A = fitensemble(TrainDataA,TrainLabelsA,'AdaBoostM1',200,'Tree','Cost',[0 1;10 0]);

[YfitB,AdaScoreB]=AdaBoostM1A.predict(TrainDataB);

AdaBoostM1B = fitensemble(TrainDataB,TrainLabelsB,'AdaBoostM1',200,'Tree','Cost',[0 1;10 0]);

[YfitA,AdaScoreA]=AdaBoostM1B.predict(TrainDataA);

AdaBoostM1 = fitensemble(TrainData,TrainLabels,'AdaBoostM1',200,'Tree','Cost',[0 1;10 0]);

[Yfit,AdaScore]=AdaBoostM1.predict(TestData);

dlmwrite('Stack3/AdaA.csv', AdaScoreA(:,2), 'delimiter', ',', 'precision', 10); 
dlmwrite('Stack3/AdaB.csv', AdaScoreB(:,2), 'delimiter', ',', 'precision', 10);
dlmwrite('Stack3/AdaTest.csv', AdaScore(:,2), 'delimiter', ',', 'precision', 10);

clear AdaScore AdaScoreB AdaScoreA Yfit YfitA YfitB AdaBoostM1 AdaBoostM1B AdaBoostM1A
%% Training KNN
disp('KNN');
knnA = fitcknn(TrainDataA,TrainLabelsA,'NumNeighbors',5,'Standardize',1,'Cost',[0 1;10 0]);

[knnYB,knnScoreB]=knnA.predict(TrainDataB);

knnB = fitcknn(TrainDataB,TrainLabelsB,'NumNeighbors',5,'Standardize',1,'Cost',[0 1;10 0]);

[knnYA,knnScoreA]=knnB.predict(TrainDataA);

knn = fitcknn(TrainData,TrainLabels,'NumNeighbors',5,'Standardize',1,'Cost',[0 1;10 0]);

[knnY,knnScore]=knn.predict(TestData);

dlmwrite('Stack3/KnnA.csv', knnScoreA(:,2), 'delimiter', ',', 'precision', 10); 
dlmwrite('Stack3/KnnB.csv', knnScoreB(:,2), 'delimiter', ',', 'precision', 10);
dlmwrite('Stack3/KnnTest.csv', knnScore(:,2), 'delimiter', ',', 'precision', 10);
%%
clear knnScore knnScoreA knnScoreB knnY knnYA knnYB knn knnA knnB
disp('finished');