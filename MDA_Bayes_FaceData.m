clc;clear;
%% Problem Parameters

% Dataset
data_folder = '../Data/';

%Test Ratio
test_ratio = 0.2;

%% Load Face Data
load([data_folder,'data.mat'])
Ns = 200;
face_n = face(:,:,1:3:3*Ns);
face_x = face(:,:,2:3:3*Ns);
face_il = face(:,:,3:3:3*Ns);

i = randi([1,Ns],1);

% Convert the dataset in data vectors and labels for
% netutral vs facil expression classification

data = [];
labels = [];
[m,n] = size(face_n(:,:,i));
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

%% Train a Bayes Classifier

% Best Estimates
Tclass_labels = unique(train_labels);
Tnum_classes = length(Tclass_labels);
Tclass_means = cell(1,Tnum_classes);
class_covariances = cell(1,Tnum_classes);
for i = 1:Tnum_classes
    class_i = (train_labels == Tclass_labels(i));
    class_data = train_data_mda(class_i,:);
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
num_data = size(test_data_mda,1);
class_probs = zeros(num_data,Tnum_classes);
for i = 1:Tnum_classes
    class_prob = class_con_probs{i}(test_data_mda);
    class_probs(:,i) = class_prob;
end
[~, predictions] = max(class_probs, [], 2);
predictions = Tclass_labels(predictions);

% Evaluate performance
acc = sum(predictions == test_labels)/numel(test_labels);