% Farinaz Fallahpour
% Date: 2011 
% https://github.com/FarinazFallahpour

%% E. Density Estimation and Classification
function DensityEstimationandClassification()
clc;
NOofFeature=2;
NOofSampel=1000;
X=CreateDataset(NOofSampel,NOofFeature);
class1=X(1:NOofSampel,:);
class2=X(NOofSampel+1:end,:);
%% a. Divide the samples into train and test sets (90% of samples as train samples and the
%     remaining 10% as test ones).
train_data1=zeros(900,3);
train_data2=zeros(900,3);
test_data=zeros(200,3);
train_data1(:,1:2)=class1(1:900,1:2);
train_data1(:,3)=0;
train_data2(:,1:2)=class2(1:900,1:2);
train_data2(:,3)=1;
train_data=zeros(1800,3);
train_data(1:900,1:2)=train_data1(:,1:2);
train_data(901:1800,1:2)=train_data2(:,1:2);
test_data(1:100,1:2)=class1(901:1000,1:2);
test_data(1:100,3)=0;
test_data(101:200,1:2)=class2(901:1000,1:2);
test_data(101:200,3)=1;
%% b. Classify the data using Bayesian method according to the estimated pdfs and report the
%     train and test accuracies of data.
%% classyfiy test data
class1i=zeros(1,2);
class2i=zeros(1,2);
k0=0,k1=0,k=1,j=1;
for i =1:200
    for n=1:100
        P1(n,:)=(1/(2*pi))* exp(-0.5*(( test_data(i,1:2)-test_data(n,1:2))*( test_data(i,1:2)-test_data(n,1:2))')/(0.1^2));   
    end
    p1(i,:)=(1/(200*(0.1^2)))*sum(P1);     
end
for i =1:200 
    for n=1:100
        P2(n,:)=(1/(2*pi))* exp(-0.5*(( test_data(i,1:2)-test_data(n+100,1:2))*( test_data(i,1:2)-test_data(n+100,1:2))')/(0.1^2));
    end
    p2(i,:)=(1/(200*(0.1^2)))*sum(P2);
end     
for i=1:200
    if p1(i,1)>p2(i,1)
        class1i(j,1:2)=test_data(i,1:2);
        class1i(j,3)=0;
        k0=k0+1;
        j=j+1;
    else
        class2i(k,1:2)=test_data(i,1:2);
        class2i(k,3)=1;
        k1=k1+1;
        k=k+1;
    end
 end
figure;
plot3(test_data(1:100,1),test_data(1:100,2),test_data(1:100,3),'ob');
hold on;
plot3(class1i(:,1),class1i(:,2),class1i(:,3),'*g');
hold on;
plot3(test_data(101:200,1),test_data(101:200,2),test_data(101:200,3),'Or');
hold on;
plot3(class2i(:,1),class2i(:,2),class2i(:,3),'*y');
title('Classify the test data');
legend('test1','classify test1','test2','classify test2');
%% classify train data
classt1=zeros(1,2);
classt2=zeros(1,2);
k0=0,k1=0,k=1,j=1;
for i =1:1800 
    for n=1:900
        Pt1(n,:)=(1/(2*pi))* exp(-0.5*(( train_data(i,1:2)-train_data1(n,1:2))*( train_data(i,1:2)-train_data1(n,1:2))')/(0.1^2));
    end
    pt1(i,:)=(1/(1800*(0.1^2)))*sum(Pt1);    
end
for i =1:1800
    for n=1:900
        Pt2(n,:)=(1/(2*pi))* exp(-0.5*(( train_data(i,1:2)-train_data2(n,1:2))*( train_data(i,1:2)-train_data2(n,1:2))')/(0.1^2));
    end
    pt2(i,:)=(1/(1800*(0.1^2)))*sum(Pt2);
end      
for i=1:1800
    if pt1(i,1)>pt2(i,1)
        classt1(j,1:2)=train_data(i,1:2);
        classt1(j,3)=0;
        k0=k0+1;
        j=j+1;
    else
    classt2(k,1:2)=train_data(i,1:2);
    classt2(k,3)=1;
    k1=k1+1;
    k=k+1;
    end
end
figure;
plot3(train_data1(:,1),train_data1(:,2),train_data1(:,3),'ob');
hold on;
plot3(classt1(:,1),classt1(:,2),classt1(:,3),'*g');
hold on;
plot3(train_data2(:,1),train_data2(:,2),train_data2(:,3),'or');
hold on;
plot3(classt2(:,1),classt2(:,2),classt2(:,3),'*y');
title('Classify the trin data');
legend('train1','classify train1','train2','classify train2');
%% c. Classify all samples using Bayesian method and compute the accuracy over all data.
%     Then, compare the results in this part with the results in part A.
class1a=zeros(1,2)
class2a=zeros(1,2);
k0=0,k1=0,k=1,j=1;
Pa1=zeros(1,1)
Pa2=zeros(1,1);
pa1=zeros(1,1);
pa2=zeros(1,1);
for i =1:2000  
    for n=1:1000
        Pa1(n,:)=(1/(2*pi))* exp(-0.5*((X(i,1:2)-class1(n,1:2))*(X(i,1:2)-class1(n,1:2))')/(0.1^2));
    end
    pa1(i,:)=(1/(200*(0.1^2)))*sum(Pa1);      
end
for i =1:2000
    for n=1:1000
        Pa2(n,:)=(1/(2*pi))* exp(-0.5*((X(i,1:2)-class2(n,1:2))*(X(i,1:2)-class2(n,1:2))')/(0.1^2));
    end
    pa2(i,:)=(1/(200*(0.1^2)))*sum(Pa2);
end    
for i=1:2000
    if pa1(i,1)>pa2(i,1)
        class1a(j,1:2)=X(i,1:2);
        class1a(j,3)=0;
        k0=k0+1;
        j=j+1;
    else
    class2a(k,1:2)=X(i,1:2);
    class2a(k,3)=1;
    k1=k1+1;
    k=k+1;
    end
end
figure;
plot3(class1(1:1000,1),class1(1:1000,2),class1(1:1000,3),'ob');
hold on;
plot3(class1a(:,1),class1a(:,2),class1a(:,3),'*g');
hold on;
plot3(class2(1:1000,1),class2(1:1000,2),class2(1:1000,3),'or');
hold on;
plot3(class2a(:,1),class2a(:,2),class2a(:,3),'*y');
title('Classify the all data');
legend('all class1','classify class1','all class2','classify class2')
end