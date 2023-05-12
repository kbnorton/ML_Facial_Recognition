clc;clear;
%% Problem Parameters

% Dataset
data_folder = '../Data/';

%Test Ratio
test_ratio = 0.8;

%% Load Face Data
load([data_folder,'data.mat'])
Ns = 200;
face_n = face(:,:,1:3:3*Ns);
face_x = face(:,:,2:3:3*Ns);
face_il = face(:,:,3:3:3*Ns);

% Convert the dataset in data vectors and labels for
% netutral vs facil expression classification

data = [];
labels = [];
[m,n] = size(face_n(:,:,1));
for subject=1:Ns
    %neutral face: label 0
    face_n_vector = reshape(face_n(:,:,subject),1,m*n);
    data = [data ; face_n_vector];
    labels = [labels 0];
    %face with expression: label 1
    face_x_vector = reshape(face_x(:,:,subject),1,m*n);
    data = [data ; face_x_vector];
    labels = [labels 1];  
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
train_data = data(idx(1:N),:);
train_labels = labels(idx(1:N));
test_data = data(idx(N+1:2*Ns),:);
test_labels = labels(idx(N+1:2*Ns));

% Perform PCA
numPCs = 20;
coeff = pca(data,'NumComponents',numPCs);
PCs = coeff(:, 1:numPCs);
train_data_pca = train_data * coeff;
test_data_pca = test_data * coeff;

%% Train a Bayes Classifier

% Best Estimates
class_labels = unique(train_labels);
num_classes = length(class_labels);
class_means = cell(1,2);
class_covariances = cell(1,2);
for i = 1:num_classes
    class_i = (train_labels == class_labels(i));
    class_data = train_data_pca(class_i,:);
    class_means{i} = mean(class_data,1);
    class_covariances{i} = cov(class_data,1) + (10^(-8))*eye(length(class_data(1,:)));
end

% Conditional probabilities
class_con_probs = cell(1, num_classes);
for i = 1:num_classes
    mean = class_means{i};
    covar = class_covariances{i};
    class_con_probs{i} = @(X) mvnpdf(X, mean, covar);
end

% Predict labels for test data
num_data = size(test_data_pca,1);
class_probs = zeros(num_data,num_classes);
for i = 1:num_classes
    class_prob = class_con_probs{i}(test_data_pca);
    class_probs(:,i) = class_prob;
end
[~, predictions] = max(class_probs, [], 2);
predictions = class_labels(predictions);

% Evaluate performance
acc = sum(predictions == test_labels)/numel(test_labels);

