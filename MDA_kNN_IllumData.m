clc;clear;
%% Problem Parameters

% Dataset
data_folder = '../Data/';

%Test Ratio
test_ratio = 0.2;

%% Load Illumination Data

load([data_folder,'illumination.mat'])
[data_size,images,subjects] = size(illum);

% Convert the datase in data vectors and labels for subject identification
data = [];
labels = [];
for s=1:subjects
    for i=1:images
        data = [data;illum(:,i,s)'];
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

%% Perform MDA analysis

% Compute Class Means
class_labels = unique(labels);
num_classes = length(class_labels);
class_means = zeros(num_classes, size(data,2));
for i = 1:num_classes
    class_i = (labels == class_labels(i));
    class_means(i, :) = mean(data(class_i, :), 1);
end

% Compute Within Class Scatter Matrix
wClass_scatter = zeros(size(data, 2));
for i = 1:num_classes
    class_i = (labels == class_labels(i));
    class_data = data(class_i, :);
    classMean = class_means(i, :);
    class_scatter = transpose((class_data - classMean)) * (class_data - classMean);
    wClass_scatter = wClass_scatter + class_scatter;
end

% Compute Between Class Scatter Matrix
Tmean = mean(data,1);
bClass_scatter = zeros(size(data,2));
for i = 1:num_classes
    class_i = (labels == class_labels(i));
    class_size = sum(class_i);
    classMean = class_means(i,:);
    bClass_scatter = bClass_scatter + class_size * transpose((classMean - Tmean)) * (classMean - Tmean);
end

% Compute FDR
[V,D] = eig(transpose(wClass_scatter)*bClass_scatter);
eigenvalues = diag(D);
[sorted_values, idx] = sort(eigenvalues,'descend');
dimensions = num_classes -1;
selected_values = sorted_values(1:dimensions);
selected_idx = idx(1:dimensions);
W = V(:, selected_idx);

% Project Data
train_data_mda = train_data * W;
test_data_mda = test_data * W;

%% Train a k-NN model

% Compute distances
all_distances = pdist2(test_data_mda, train_data_mda, 'euclidean');

% Find k Nearest Neighbors
k = 8 ;
[distances, indices] = mink(all_distances, k, 2);

% Predict labels for test data
weighted_votes = zeros(length(test_data_mda),length(unique(labels)));
weights = 1 ./ (distances .^2);
votes = train_labels(indices);
for i = 1:length(test_data_mda)
    for j = 1:k
        weighted_votes(i,votes(i,j)) = weighted_votes(i,votes(i,j)) + weights(i,j);
    end
end
[~, predictions] = max(transpose(weighted_votes));

% Evaluate Performance
acc = sum((predictions) == test_labels)/numel(test_labels);
