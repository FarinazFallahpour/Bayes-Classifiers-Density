% Farinaz Fallahpour
% Date: 2011 
% https://github.com/FarinazFallahpour

%% A. Bayesian classifier
% a. Generate a dataset of 2000 samples (1000 random samples from each of the classes).
function Dataset=CreateDataset(NOofSampel,NOofFeature)
clc;
%% tolide adade random
Dataset1=randn(NOofSampel,NOofFeature);
Dataset2=randn(NOofSampel,NOofFeature);
%% class0 with gaussian distributed with mean[-1,1] and label=0
for i=1:NOofSampel
    Dataset1(i,1)=Dataset1(i,1)-1;
    Dataset1(i,2)=Dataset1(i,2)+1;
end
Dataset1=[Dataset1,zeros(NOofSampel,1)];
%% class1 with gaussian distributed with mean[1,0] and label=1
for i=1:NOofSampel
    Dataset2(i,1)=Dataset2(i,1)+1;
    Dataset2(i,2)=Dataset2(i,2)+0;
end
Dataset2=[Dataset2,ones(NOofSampel,1)];
%% dataset inclusive class0 and class1
for i=1:NOofSampel
    for j=1:NOofFeature+1
        Dataset(i,j)=Dataset1(i,j);
    end
end
for i=1:NOofSampel
    for j=1:NOofFeature+1
        Dataset(i+NOofSampel,j)=Dataset2(i,j);
    end
end
end