% Farinaz Fallahpour
% Date: 2011 
% https://github.com/FarinazFallahpour

%% A. Bayesian classifier
function BayesClassifier()
clc;
%% a. Generate a dataset of 2000 samples (1000 random samples from each of the classes).
NOofFeature=2;
NOofSampel=1000;
X=CreateDataset(NOofSampel,NOofFeature);
%% plot two classes without labels
X1=X(:,1:2);
figure;
plot(X1(1:NOofSampel,1),X1(1:NOofSampel,2),'.b');
hold on;
plot(X1(NOofSampel+1:end,1),X1(NOofSampel+1:end,2),'.r');
xlabel('Feature1');
ylabel('Feature2');
legend('Samples from Class0','Samples from Class1');
title('Data');
%% plot two classes with labels
X2=X(:,1:3);
figure;
plot3(X2(1:NOofSampel,1),X2(1:NOofSampel,2),X2(1:NOofSampel,3),'.b');
hold on;
plot3(X2(NOofSampel+1:end,1),X2(NOofSampel+1:end,2),X2(NOofSampel+1:end,3),'.r');
xlabel('Feature1');
ylabel('Feature2');
zlabel('Label');
legend('Samples from Class0','Samples from Class1');
title('Data with label');
%% b. Design a Bayes classifier on the training set and report the per-class accuracy of your
%     classifier on the training samples.
%% LDA
sigma0=[1 0;0 1];
sigma1=[2 2;2 3];
mean0=[-1,1];
mean1=[1,0];
% probability x given y=0
for i=1:NOofSampel
    Prob0=(1/((sqrt(2*pi))^NOofFeature)*((det(sigma0))^(1/2)))*exp(-(1/2)*(X(i,1:2)-mean0)*(inv(sigma0))*(X(i,1:2)-mean0)');
    P0(i,1)=Prob0*(1/2);
end
% probability x given y=1
for i=1001:2000
    Prob1=(1/((sqrt(2*pi))^NOofFeature)*((det(sigma1))^(1/2)))*exp(-(1/2)*(X(i,1:2)-mean1)*(inv(sigma1))*(X(i,1:2)-mean1)');
    P1(i-1000,1)=Prob1*(1/2);
end
% plot two classes with labels
figure;
plot3(X2(1:NOofSampel,1),X2(1:NOofSampel,2),P0,'.b');
hold on;
plot3(X2(NOofSampel+1:end,1),X2(NOofSampel+1:end,2),P1,'.r');
xlabel('Feature1');
ylabel('Feature2');
zlabel('Posterior');
legend('Samples from Class0','Samples from Class1');
title('Class Shape');
%% accuracy
Accuracy=0;
Prob=zeros(1,NOofSampel);
for i=1:NOofSampel
    if P1(i)> P0(i)
        Prob(i)=1;
    end
    if Prob(i)==X(i,3)
        Accuracy=Accuracy+1;
    end
end
disp('Accuracy:');
disp(Accuracy);
disp('NoofSampel:');
disp(NOofSampel);
%% a. Derive mathematically the true boundary of two classes. Sketch the dataset and the true
%     Bayes discriminant boundary of two classes.
%% QDA
INVsigma0 = inv(sigma0);
INVsigma1 = inv(sigma1);
% Calculate boundary
w0 = -0.5*log(det(sigma0)) + 0.5*log(det(sigma1)) - 0.5*mean0*INVsigma0*mean0'+ 0.5*mean1*INVsigma1*mean1';
w1 = mean0*INVsigma0 - mean1*INVsigma1;
w2 = INVsigma0 - INVsigma1;
% Calculate log-likelihood 
Xt=X(:,1:2)';
for i=1:2000
   l(i) = -0.5*Xt(:,i)'*w2*Xt(:,i)+w1*Xt(:,i)+w0;
end
%% boundary
X1=X(:,1:2);
figure;
plot(X1(1:NOofSampel,1),X1(1:NOofSampel,2),'.b');
hold on;
plot(X1(NOofSampel+1:end,1),X1(NOofSampel+1:end,2),'.r');
hold on;
plot_x = [min(X(:,2))-2, max(X(:,2))+2];
plot_y = (1./l(3)).*(l(2).*plot_x +l(1));
plot(plot_x, plot_y,'-g');
xlabel('Feature1');
ylabel('Feature2');
legend('Samples from Class0','Samples from Class1','Decision Boundary');
title('Data');
%% b. In most practical applications the cost of misclassification is not equal for all classes. For
%     example, miss-classifying a person who has cancer as a healthy one (False Negative) is
%     much more expensive than misclassifying a person who doesn’t have cancer (False
%     Positive). Suppose that the cost of misclassifying a sample from class 1 is C times greater
%     than the one from class 2. It means that !(!"#$=1|!)>!×!(!"#$=2 |!). Derive the
%     Bayes classifier for C= 1, 2, …, 10 and plot the accuracy (total accuracy, accuracy of each
%     class) in terms of different C values.
end