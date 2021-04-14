% Farinaz Fallahpour
% Date: 2011 
% https://github.com/FarinazFallahpour
%% B. Naïve Bayesian Classifier
function NaiveBayesian()
%---------------Train-------------------------
%% Defenition & Load  Train Dataset
clc;
% M= load('German2_Train.txt');Lable1=1;Lable2=2;
 M = dlmread('Thalassemi_Train.txt'); Lable1=-1;Lable2=1;
train_matrix=M(:,1:end-1);
[NoOfTrainDocs,NoOfFeature]=size(train_matrix);
train_labels = M(:,end);
%% Find the indices for the Class1 and Class2 labels
Class1_indices = find(train_labels==Lable1);
Class2_indices = find(train_labels==Lable2);
%% Calculate probability of Class1
Prob_Class1 = length(Class1_indices) / NoOfTrainDocs;
Lengths= sum(train_matrix, 2);
%% find the total  counts 
Class1_wc = sum(Lengths(Class1_indices));
Class2_wc = sum(Lengths(Class2_indices));
%% Calculate the probability of the tokens in Class1
prob_tokens_Class1 = (sum(train_matrix(Class1_indices, :)) + 1) ./ ...
    (Class1_wc + NoOfFeature);
%The k-th entry of prob_tokens_Class1 represents phi_(k|y=Lable1)
%% Calculate the probability of the tokens in Class2
prob_tokens_Class2 = (sum(train_matrix(Class2_indices, :)) + 1)./ ...
    (Class2_wc + NoOfFeature);
%The k-th entry of prob_tokens_nonspam represents phi_(k|y=Lable2)
%% -----------------------Test----------------------
% N=load('German2_Test.txt');
 N=load('Thalassemi_Test.txt');
test_matrix=N(:,1:end-1);
NoOfTestDocs = size(test_matrix,1);
% Calculate log p(x|y=1) + log p(y=1)
% and log p(x|y=0) + log p(y=0)
% for every document
% make your prediction based on what value is higher
log_a = test_matrix*(log(prob_tokens_Class1))' + log(Prob_Class1);
log_b = test_matrix*(log(prob_tokens_Class2))'+ log(1 - Prob_Class1);  
for i=1:NoOfTestDocs
    if (log_a(i) > log_b(i))
    disp(Lable1);
else disp(Lable2);
    end
end
%% Accuracy
log_a_train= train_matrix*(log(prob_tokens_Class1))' + log(Prob_Class1);
log_b_train = train_matrix*(log(prob_tokens_Class2))'+ log(1 - Prob_Class1); 
output=zeros(1,NoOfTrainDocs);
for i=1:NoOfTrainDocs
    if (log_a_train(i) > log_b_train(i))
    output(i)=Lable1;
    else 
    output(i)=Lable2;
    end
end
TruePredict=sum(output==train_labels');
 Accuracy=(TruePredict/NoOfTrainDocs);
 disp(Accuracy);
end