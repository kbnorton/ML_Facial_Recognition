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
numPCs = 25;
coeff = pca(data,'NumComponents',numPCs);
PCs = coeff(:, 1:numPCs);
train_data_pca = train_data * coeff;
test_data_pca = test_data * coeff;

%% Train a k-NN model

% Compute distances
distances = pdist2(test_data_pca, train_data_pca, 'euclidean');

% Find k Nearest Neighbors
k = 15;
[~, indices] = mink(distances, k, 2);

% Predict labels for test data
predictions = mode(train_labels(indices),2);

% Evaluate Performance
acc = sum(transpose(predictions) == test_labels)/numel(test_labels);
