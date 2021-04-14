% Farinaz Fallahpour
% Date: 2011 
% https://github.com/FarinazFallahpour

%% C. Parametric Density estimation
function ParametricDensityEstimation()
clc;
NOofFeature=2;
NOofSampel=1000;
X=CreateDataset(NOofSampel,NOofFeature);
%% a. Estimate the mean and covariance of each class using Maximum Likelihood approach.
% parameters of this model:(phi,mean0,mean1,sigma0,sigma1)
%% mean0
for i=1:NOofSampel
    mean0=(sum(Indicator(X(i,3)))*X(i,1:2))/(sum(Indicator(X(i,3))));
end
disp('mean0:');
disp(mean0);
%% mean1
for i=NOofSampel+1:2000
    mean1=(sum(Indicator(X(i,3)))*X(i,1:2))/(sum(Indicator(X(i,3))));
end
disp('mean1:');
disp(mean1);
%% sigma0
sigma0=zeros(2,2);
for i=1:NOofSampel
    sigma0=sigma0+((X(i,1:2)-mean0)'*(X(i,1:2)-mean0));
end
sigma0=(1/NOofSampel).*sigma0;
disp('sigma0:');
disp(sigma0);
%% sigma1
sigma1=zeros(2,2);
for i=NOofSampel+1:2000
    sigma1=sigma1+((X(i,1:2)-mean1)'*(X(i,1:2)-mean1));
end
sigma1=(1/NOofSampel).*sigma1;
disp('sigma1:');
disp(sigma1);
%% b. Plot the estimated pdfs and true pdfs and compare them.
%% estimated pdf
% probability x given y=0
for i=1:NOofSampel
    EProb0=(1/((sqrt(2*pi))^NOofFeature)*((det(sigma0))^(1/2)))*exp(-(1/2)*(X(i,1:2)-mean0)*(inv(sigma0))*(X(i,1:2)-mean0)');
    EP0(i,1)=EProb0*(1/2);
end
% probability x given y=1
for i=1001:2000
    EProb1=(1/((sqrt(2*pi))^NOofFeature)*((det(sigma1))^(1/2)))*exp(-(1/2)*(X(i,1:2)-mean1)*(inv(sigma1))*(X(i,1:2)-mean1)');
    EP1(i-1000,1)=EProb1*(1/2);
end
% plot two classes with labels
figure;
plot3(X(1:NOofSampel,1),X(1:NOofSampel,2),EP0,'.g');
hold on;
plot3(X(NOofSampel+1:end,1),X(NOofSampel+1:end,2),EP1,'.y');
xlabel('Feature1');
ylabel('Feature2');
zlabel('Posterior');
legend('Class0','Class1');
title('Estimated PDF');
%% true pdf
sigma0=[1 0;0 1];
sigma1=[2 2;2 3];
mean0=[-1,1];
mean1=[1,0];
% probability x given y=0
for i=1:NOofSampel
    TProb0=(1/((sqrt(2*pi))^NOofFeature)*((det(sigma0))^(1/2)))*exp(-(1/2)*(X(i,1:2)-mean0)*(inv(sigma0))*(X(i,1:2)-mean0)');
    TP0(i,1)=TProb0*(1/2);
end
% probability x given y=1
for i=1001:2000
    TProb1=(1/((sqrt(2*pi))^NOofFeature)*((det(sigma1))^(1/2)))*exp(-(1/2)*(X(i,1:2)-mean1)*(inv(sigma1))*(X(i,1:2)-mean1)');
    TP1(i-1000,1)=TProb1*(1/2);
end
% plot two classes with labels
figure;
plot3(X(1:NOofSampel,1),X(1:NOofSampel,2),TP0,'.b');
hold on;
plot3(X(NOofSampel+1:end,1),X(NOofSampel+1:end,2),TP1,'.r');
xlabel('Feature1');
ylabel('Feature2');
zlabel('Posterior');
legend('Class0','Class1');
title('True PDF');
%% compare pdf
figure;
plot3(X(1:NOofSampel,1),X(1:NOofSampel,2),EP0,'.g');
hold on;
plot3(X(NOofSampel+1:end,1),X(NOofSampel+1:end,2),EP1,'.y');
hold on;
plot3(X(1:NOofSampel,1),X(1:NOofSampel,2),TP0,'.b');
hold on;
plot3(X(NOofSampel+1:end,1),X(NOofSampel+1:end,2),TP1,'.r');
xlabel('Feature1');
ylabel('Feature2');
zlabel('Posterior');
legend('Estimate Class0','Estimate Class1','True Class0','True Class1');
title('Compare PDFs');
end