clc; clear;
%% Problem Parameters

% Dataset
data_folder = '../Data/';

%Test Ratio
test_ratio = 0.2;

%% Load Pose Data

load([data_folder,'pose.mat'])
[rows,columns,images,subjects]= size(pose);

% Convert the datase in data vectors and labels for subject identification
data = [];
labels = [];
for s=1:subjects
    for i=1:images
        pose_vector = reshape(pose(:,:,i,s),1,rows*columns);
        data = [data;pose_vector];
        labels = [labels s];        
    end
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
train_data = data(idx(1:N),:);
train_labels = labels(idx(1:N));
test_data = data(idx(N+1:data_len),:);
test_labels = labels(idx(N+1:data_len));

% Perform PCA
numPCs = 30;
coeff = pca(data,'NumComponents',numPCs);
PCs = coeff(:, 1:numPCs);
train_data_pca = train_data * coeff;
test_data_pca = test_data * coeff;

%% Train a Bayes Classifier

% Best Estimates
Tclass_labels = unique(train_labels);
Tnum_classes = length(Tclass_labels);
Tclass_means = cell(1,Tnum_classes);
class_covariances = cell(1,Tnum_classes);
for i = 1:Tnum_classes
    class_i = (train_labels == Tclass_labels(i));
    class_data = train_data_pca(class_i,:);
    Tclass_means{i} = mean(class_data,1);
    class_covariances{i} = cov(class_data,1) + (10^(-8))*eye(length(class_data(1,:)));
end

% Conditional probabilities
class_con_probs = cell(1, Tnum_classes);
for i = 1:Tnum_classes
    mean = Tclass_means{i};
    covar = class_covariances{i};
    class_con_probs{i} = @(X) mvnpdf(X, mean, covar);
end

% Predict labels for test data
num_data = size(test_data_pca,1);
class_probs = zeros(num_data,Tnum_classes);
for i = 1:Tnum_classes
    class_prob = class_con_probs{i}(test_data_pca);
    class_probs(:,i) = class_prob;
end
[~, predictions] = max(class_probs, [], 2);
predictions = Tclass_labels(predictions);

% Evaluate performance
acc = sum(predictions == test_labels)/numel(test_labels);