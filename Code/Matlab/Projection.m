clear all; close all; clc;
addpath('./tools');

DATASET_PATH = './dataSet/';
NB_CAMERA = 2; NB_PERSON = 9; 
NB_VECTOR = 4; IMG_SIZE  = [30,30];


%% Load the image of Camera %%
image_selected = [DATASET_PATH, 'ImagesSelected/'];
camera = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        filePattern = fullfile([image_selected, 'video',int2str(i),'/p',int2str(j)], '/*.png');
        pngFiles = dir(filePattern);
        % PNG Files except the mask
        pngFiles = pngFiles(~contains({pngFiles.name}, 'mask'));
        camera{i,j} = ones(IMG_SIZE(1) * IMG_SIZE(2), length(pngFiles));
        for k = 1:length(pngFiles)
            baseFileName = pngFiles(k).name;
            fullFileName = fullfile(image_selected, ['video',int2str(i),'/p',int2str(j),'/', baseFileName]);
            camera{i,j}(:,k) = reshape(double(imread(fullFileName)),IMG_SIZE(1) * IMG_SIZE(2), 1);
        end
    end
end


%% Compute the Mean Face %%
mean_target = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        mean_target{i,j} = transpose(mean(double(transpose(camera{i,j}))));
    end
end

%% Compute the Zero Mean of the Images %%
zero_mean = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        for k = 1:size(camera{i,j},2)
            zero_mean{i,j}(:,k) = camera{i,j}(:,k) - mean_target{i,j};
        end
    end
end


%% Compute the Eigentarget %%
pcaEigVal = cell(NB_CAMERA,NB_PERSON);
pcaEigVec = cell(NB_CAMERA,NB_PERSON);

for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        [pcaEigVal{i,j},pcaEigVec{i,j}] = Turk_Pentland(zero_mean{i,j},NB_VECTOR);
    end
end


% Plot one person one camera, mean face, zero mean and Eigentarget
c1p1 = camera{1,1};
figure;
nFrames = size(c1p1,2);
for i = 1:nFrames
    subplot(nFrames, 4,4*(i-1)+1);
    imshow(mat2gray(reshape(c1p1(:,i),IMG_SIZE(1),IMG_SIZE(2))))
    if i == 1
        title('Original Image');
    end
end

subplot(nFrames, 4, 2);
imshow(mat2gray(reshape(mean_target{1,1},IMG_SIZE(1),IMG_SIZE(2))));
title('Mean Target');

for i = 1:nFrames
    subplot(nFrames, 4, 4*(i-1)+3);
    imshow(mat2gray(reshape(zero_mean{1,1}(:,i),IMG_SIZE(1),IMG_SIZE(2))));
    if i == 1
        title('Zero Mean');
    end
end

for i = 1:size(pcaEigVec{1,1},2)
    subplot(nFrames, 4, 4*(i-1)+4);
    imshow(mat2gray(reshape(pcaEigVec{1,1}(:,i),IMG_SIZE(1),IMG_SIZE(2))));
    if i == 1
        title('Eigentarget');
    end
end