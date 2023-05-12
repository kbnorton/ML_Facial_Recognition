clc;clear;
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
numPCs = 25;
coeff = pca(data,'NumComponents',numPCs);
PCs = coeff(:, 1:numPCs);
train_data_pca = train_data * coeff;
test_data_pca = test_data * coeff;

%% Train a k-NN model

% Compute distances
all_distances = pdist2(test_data_pca, train_data_pca, 'euclidean');

% Find k Nearest Neighbors
k = 8;
[distances, indices] = mink(all_distances, k, 2);

% Predict labels for test data
weighted_votes = zeros(length(test_data_pca),length(unique(labels)));
weights = 1 ./ (distances .^2);
votes = train_labels(indices);
for i = 1:length(test_data_pca)
    for j = 1:k
        weighted_votes(i,votes(i,j)) = weighted_votes(i,votes(i,j)) + weights(i,j);
    end
end
[~, predictions] = max(transpose(weighted_votes));

% Evaluate Performance
acc = sum((predictions) == test_labels)/numel(test_labels);
