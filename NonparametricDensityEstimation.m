% Farinaz Fallahpour
% Date: 2011 
% https://github.com/FarinazFallahpour
% D. Non-parametric Density Estimation
function NonparametricDensityEstimation()
%Histogram();
%GaussianKernel();
KNNEstimator();
%% a. Estimate the pdfs using the Histogram method with bin width 0.1 and 0.5. Plot the
%     estimated pdfs versus the true pdfs and explain the effect of bin width on the estimated
%     pdfs.
function Histogram()
clc;
NOofBin=0;
NOofFeature=2;
NOofSampel=1000;
X=CreateDataset(NOofSampel,NOofFeature);
%% histogram class0 for feature1
Class0=X(1:NOofSampel,1);
minClass0=min(Class0);
maxClass0=max(Class0);
minClass0i=minClass0;
maxClass0i=maxClass0;
NOofDatainBin=0;
while minClass0<maxClass0
    for i=1:NOofSampel
        if (minClass0<=Class0(i,1)) && (Class0(i,1)<=(minClass0+0.1))
            NOofDatainBin=NOofDatainBin+1;
        end
    end
    minClass0=minClass0+0.1;
    NOofBin=NOofBin+1;
    Data(NOofBin)=NOofDatainBin;
    NOofDatainBin=0;
end
    for j=1:NOofBin
        P(j)=(1/NOofSampel)*(Data(j)/0.1);
        P(j);
    end
figure;
Bin=minClass0i:0.1:maxClass0i;
bar(Bin,Data);
title('Histogram Class0 for Feature1');
%% histogram class0 for feature2
NOofBin1=0;
Class01=X(1:NOofSampel,2);
minClass01=min(Class01);
maxClass01=max(Class01);
minClass0i1=minClass01;
maxClass0i1=maxClass01;
NOofDatainBin1=0;
while minClass01<maxClass01
    for i=1:NOofSampel
        if (minClass01<=Class01(i,1)) && (Class01(i,1)<=(minClass01+0.1))
            NOofDatainBin1=NOofDatainBin1+1;
        end
    end
    minClass01=minClass01+0.1;
    NOofBin1=NOofBin1+1;
    Data1(NOofBin1)=NOofDatainBin1;
    NOofDatainBin1=0;
end
    for j=1:NOofBin1
        P(j)=(1/NOofSampel)*(Data1(j)/0.1);
        P(j);
    end
figure;
Bin1=minClass0i1:0.1:maxClass0i1;
bar(Bin1,Data1);
title('Histogram Class0 for Feature2');
%% histogram class1 for feature1
NOofBin=0;
Class1=X(NOofSampel+1:end,1);
minClass1=min(Class1);
maxClass1=max(Class1);
minClass1i=minClass1;
maxClass1i=maxClass1;
NOofDatainBin=0;
while minClass1<maxClass1
    for i=1:NOofSampel
        if (minClass1<=Class1(i,1)) && (Class1(i,1)<=(minClass1+0.5))
            NOofDatainBin=NOofDatainBin+1;
        end
    end
    minClass1=minClass1+0.5;
    NOofBin=NOofBin+1;
    Data2(NOofBin)=NOofDatainBin;
    NOofDatainBin=0;
end
  
for j=1:NOofBin
    P(j)=(1/1000)*(Data2(j)/0.5);
    P(j);
end
figure;
Bin2=minClass1i:0.5:maxClass1i;
bar(Bin2,Data2);
title('Histogram Class1 for Feature1');
%% histogram class1 for feature2
NOofBin=0;
Class1=X(NOofSampel+1:end,2);
minClass1=min(Class1);
maxClass1=max(Class1);
minClass1i=minClass1;
maxClass1i=maxClass1;
NOofDatainBin=0;
while minClass1<maxClass1
    for i=1:NOofSampel
        if (minClass1<=Class1(i,1)) && (Class1(i,1)<=(minClass1+0.5))
            NOofDatainBin=NOofDatainBin+1;
        end
    end
    minClass1=minClass1+0.5;
    NOofBin=NOofBin+1;
    Data3(NOofBin)=NOofDatainBin;
    NOofDatainBin=0;
end
  
for j=1:NOofBin
    P(j)=(1/1000)*(Data3(j)/0.5);
    P(j);
end
figure;
Bin3=minClass1i:0.5:maxClass1i;
bar(Bin3,Data3);
title('Histogram Class1 for Feature2');
%% 
subplot(2,2,1),bar(Bin,Data);
title('Histogram Class0 for Feature1');
subplot(2,2,2),bar(Bin1,Data1);
title('Histogram Class0 for Feature2');
subplot(2,2,3),bar(Bin2,Data2);
title('Histogram Class1 for Feature1');
subplot(2,2,4),bar(Bin3,Data3);
title('Histogram Class1 for Feature2');
end
%% b. Estimate the pdfs with Gaussian kernel with standard deviations of 0.1, 0.25, 0.5 and 0.75.
%     Plot the estimated pdfs versus the true pdfs and explain the effect of standard deviation on
%     the estimated pdfs.
function GaussianKernel()
clc;
sigma1=[1,0;0,1];
sigma2=[2,2;2,3];
mean1=[-1,1];
mean2=[1,0];
NOofFeature=2;
NOofSampel=1000;
X=CreateDataset(NOofSampel,NOofFeature);
class1=X(1:NOofSampel,:);
class2=X(NOofSampel+1:end,:);
%% bayes classifier
p1=zeros(1,1);
p2=zeros(1,1);
class1i=zeros(1,2)
class2i=zeros(1,2);
k0=0,k1=0,k=1,j=1;
for i =1:2000  
    p1(i,:)=(1/((sqrt(2*pi))^2*det(sigma1)^1/2))* exp(-1/2* (X(i,1:2)-mean1)*inv(sigma1)*( X(i,1:2)-mean1)')*1/2;    
    p2(i,:)=(1/((sqrt(2*pi))^2*det(sigma2)^1/2))* exp(-1/2* (X(i,1:2)-mean2)*inv(sigma2)*( X(i,1:2)-mean2)')*1/2;
end      
 for i=1:2000
       if p1(i,1)>p2(i,1)
           class1i(j,1:2)=X(i,1:2);
           class1i(j,3)=0;
           k0=k0+1;
           j=j+1;
        else
           class2i(k,1:2)=X(i,1:2);
           class2i(k,3)=1;
           k1=k1+1;
           k=k+1;
        end
 end
figure;
plot3(X(1:NOofSampel,1),X(1:NOofSampel,2),X(1:NOofSampel,3),'ob');
hold on;
plot3(class1i(:,1),class1i(:,2),class1i(:,3),'*g');
title('Classify the all data according to the estimated pdfs');
hold on;
plot3(X(NOofSampel+1:end,1),X(NOofSampel+1:end,2),X(NOofSampel+1:end,3),'or');
hold on;
plot3(class2i(:,1),class2i(:,2),class2i(:,3),'*y');
title('Classify data');
legend( 'class1','Bayes classifier class1','class2','Bayes classifier class2')
%%
for i =1:NOofSampel
    P1(i,:)=(1/((sqrt(2*pi))^2*det(sigma1)^1/2))* exp(-1/2* (class1(i,1:2)-mean1)*inv(sigma1)*( class1(i,1:2)-mean1)')*0.5;
end
figure;
plot3(class1(:,1),class1(:,2),P1,'.b');
hold on
for i =1:NOofSampel
    P2(i,:)=(1/((sqrt(2*pi))^2*det(sigma2)^1/2))* exp(-1/2* (class2(i,1:2)-mean2)*inv(sigma2)*( class2(i,1:2)-mean2)')*0.5;
end
plot3(class2(:,1),class2(:,2),P2,'.r');
legend('class1','class2');
Estimate(0.1,'Class1 with standard deviations 0.1','Class2 with standard deviations 0.1');
Estimate(0.25,'Class1 with standard deviations 0.25','Class2 with standard deviations 0.25');
Estimate(0.5,'Class1 with standard deviations 0.5','Class2 with standard deviations 0.5');
Estimate(0.75,'Class1 with standard deviations 0.75','Class2 with standard deviations 0.75');
%% Estimate the pdfs with Gaussian kernel with standard deviations of 0.1
function Estimate(standardDeviation,title1,title2)
for i =1:NOofSampel
    for j=1:NOofSampel
        K(j,:)=(1/(2*pi))* exp(-0.5*(( class1(i,1:2)-class1(j,1:2))*( class1(i,1:2)-class1(j,1:2))')/(standardDeviation^2));
    end
    pd1(i,:)=(1/(2000*(0.1^2)))*sum(K);
end
figure;
subplot(1,2,1),plot3(class1(:,1),class1(:,2),P1,'.b');
hold on,plot3(class1(:,1),class1(:,2),pd1,'.g');
legend('class1','pdf');
title(title1);
for i =1:NOofSampel
    for j=1:NOofSampel
        K(j,:)=(1/(2*pi))* exp(-0.5*(( class2(i,1:2)-class2(j,1:2))*( class2(i,1:2)-class2(j,1:2))')/(standardDeviation^2));
    end
pd2(i,:)=(1/(2000*(0.1^2)))*sum(K);
end
subplot(1,2,2),plot3(class2(:,1),class2(:,2),P2,'.r')
hold on;
plot3(class2(:,1),class2(:,2),pd2,'.y');
legend('class2','pdf');
title(title2);
end
end
%% c. Do the same work of part a. and b. using KNN estimator with k=1,3,5. Plot the estimated
%     pdfs versus the true pdfs and explain the effect of k on the estimated pdfs.
function KNNEstimator()
    clc;
NOofFeature=2;
NOofSampel=1000;
X=CreateDataset(NOofSampel,NOofFeature);
class1=X(1:NOofSampel,1:2);
class2=X(NOofSampel+1:2000,1:2);
%%
k=5;
numclass1=size(class1,1);
numclass2=size(class2,1);
totalSamples=numclass1+numclass2;
combinedSamples=[class1;class2];
trueclass=[zeros(numclass1,1)+1 ;zeros(numclass2,1)+2];
for help=1:totalSamples
    newSample=combinedSamples(help,:);
    testmatrix=repmat(newSample,totalSamples,1);
    absdiff=abs(combinedSamples-testmatrix);
    absdiff=absdiff.^2;
    dist=sum(absdiff,2);
    [Y,I]=sort(dist);
    neighborsInd=I(1:k);
    neighbors=trueclass(neighborsInd);
    clas1=0;
    clas2=0;
    for temp=1:size(neighbors)
        if neighbors(temp,1)==1
            clas1=clas1+1;
        else 
            clas2=clas2+1;
        end
    end
    time=1;
    if clas2>=clas1
        for temp=1:size(neighbors)
            if neighbors(temp,1)==2 && time==1
                R=neighborsInd(temp);
                time=time+1;
            end
        end
    end
    if clas1>=clas2
        for temp=1:size(neighbors)
            if neighbors(temp,1)==1 && time==1
                R=neighborsInd(temp);
                time=time+1;
            end
        end
    end
    PPX(help,1)=k/(pi*totalSamples*R*R);
end
figure,
plot(combinedSamples(1:2000,:),PPX(1:2000,1),'g.');
legend('k==5');
%%
k=1;
numclass1=size(class1,1);
numclass2=size(class2,1);
totalSamples=numclass1+numclass2;
combinedSamples=[class1;class2];
trueclass=[zeros(numclass1,1)+1 ;zeros(numclass2,1)+2];
for help=1:totalSamples
    newSample=combinedSamples(help,:);
    testmatrix=repmat(newSample,totalSamples,1);
    absdiff=abs(combinedSamples-testmatrix);
    absdiff=absdiff.^2;
    dist=sum(absdiff,2);
    [Y,I]=sort(dist);
    neighborsInd=I(1:k);
    neighbors=trueclass(neighborsInd);
    clas1=0;
    clas2=0;
    for temp=1:size(neighbors)
        if neighbors(temp,1)==1
            clas1=clas1+1;
        else 
            clas2=clas2+1;
        end
    end
    time=1;
    if clas2>=clas1
        for temp=1:size(neighbors)
            if neighbors(temp,1)==2 && time==1
                R=neighborsInd(temp);
                time=time+1;
            end
        end
    end
    if clas1>=clas2
        for temp=1:size(neighbors)
            if neighbors(temp,1)==1 && time==1
                R=neighborsInd(temp);
                time=time+1;
            end
        end
    end
    PPX(help,1)=k/(pi*totalSamples*R*R);
end
figure,
plot(combinedSamples(1:2000,:),PPX(1:2000,1),'b.');
legend('k==1');
%%
k=3;
numclass1=size(class1,1);
numclass2=size(class2,1);
totalSamples=numclass1+numclass2;
combinedSamples=[class1;class2];
trueclass=[zeros(numclass1,1)+1 ;zeros(numclass2,1)+2];
for help=1:totalSamples
    newSample=combinedSamples(help,:);
    testmatrix=repmat(newSample,totalSamples,1);
    absdiff=abs(combinedSamples-testmatrix);
    absdiff=absdiff.^2;
    dist=sum(absdiff,2);
    [Y,I]=sort(dist);
    neighborsInd=I(1:k);
    neighbors=trueclass(neighborsInd);
    clas1=0;
    clas2=0;
    for temp=1:size(neighbors)
        if neighbors(temp,1)==1
            clas1=clas1+1;
        else 
            clas2=clas2+1;
        end
    end
    time=1;
    if clas2>=clas1
        for temp=1:size(neighbors)
            if neighbors(temp,1)==2 && time==1
                R=neighborsInd(temp);
                time=time+1;
            end
        end
    end
    if clas1>=clas2
        for temp=1:size(neighbors)
            if neighbors(temp,1)==1 && time==1
                R=neighborsInd(temp);
                time=time+1;
            end
        end
    end
    PPX(help,1)=k/(pi*totalSamples*R*R);
end
figure;
plot(combinedSamples(1:2000,:),PPX(1:2000,1),'r.');
legend('k==3');

end
end