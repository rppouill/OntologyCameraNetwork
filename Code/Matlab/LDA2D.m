clear all; close all; clc;
warning("off","all");
addpath('./tools'); addpath('./Transformation'); 
addpath('./Distance');
addpath('./Transformation/Temporal');

DATASET_PATH = './dataSet/';

%% Credit with Disp %%
disp('Files     : 2D-LDA');
disp('Author    : Romain POUILLARD');
disp('Date      : 2025-01-28');
disp('Version   : 1.0');


NB_CAMERA =  3; NB_PERSON = 9;
NB_VECTOR =  1;
IMG_SIZE = [30,30];

%% ----------------- Load the image of Camera ----------------- %%

image_selected = [DATASET_PATH, 'ImagesSelected-Rearange/'];
%image_selected = [DATASET_PATH, 'NewDataset/'];
%image_selected = [DATASET_PATH, 'SimulatorDataset/'];

if exist([image_selected, 'D.mat'], 'file')
    load([image_selected, 'D.mat']);
    D = cellfun(@(x) normalize(x, 'range'), D, 'UniformOutput', false); 
else
    D = cell(NB_CAMERA, NB_PERSON);
    for i = 1:NB_CAMERA
        for j = 1:NB_PERSON
            D{i,j} = ones(IMG_SIZE(1) * IMG_SIZE(2), 1);
        end
    end
end

camera = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        filePattern = fullfile([image_selected, 'video',int2str(i),'/p',int2str(j)], '/*.png');
        pngFiles = dir(filePattern);
        % PNG Files except the mask
        pngFiles = pngFiles(~contains({pngFiles.name}, 'mask'));
        %camera{i,j} = ones(IMG_SIZE(1) * IMG_SIZE(2), length(pngFiles));
        camera{i,j} = ones(IMG_SIZE(1), IMG_SIZE(2), length(pngFiles));
        for k = 1:length(pngFiles)
            baseFileName = pngFiles(k).name;
            fullFileName = fullfile(image_selected, ['video',int2str(i),'/p',int2str(j),'/', baseFileName]);

            %camera{i,j}(:,k) = reshape(imread(fullFileName), IMG_SIZE(1) * IMG_SIZE(2), 1);
            camera{i,j}(:,:,k) = imresize(imread(fullFileName), IMG_SIZE);
            % Normalize the image
            %camera{i,j}(:,k) = camera{i,j}(:,k) / 255;        
            camera{i,j}(:,:,k) = double(camera{i,j}(:,:,k)) / 255.0;
        end
    end
end

pcaEig = buildEvent(camera, NB_VECTOR, D);
% Reshape %
dataSet = cell(NB_CAMERA, NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        dataSet{i,j} = reshape(pcaEig{i,j}, IMG_SIZE(1), IMG_SIZE(2), []);
    end
end

% Train and Test %
%TRAIN_Ratio = 0.5; 
%dataTrain = cell(NB_CAMERA, 1); dataTest = cell(NB_CAMERA, 1);
%for i = 1:NB_CAMERA
%    for j = 1:NB_PERSON
%        N_TRAIN = round(size(dataSet{i},3) * TRAIN_Ratio);
%        N_TEST  = size(dataSet{i},3) - N_TRAIN;
%        dataTrain{i} = cat(3, dataTrain{i}, dataSet{i}(:,:,1:N_TRAIN));
%        dataTest{i}  = cat(3, dataTest{i}, dataSet{i}(:,:,N_TRAIN+1:end));
%    end
%end

TRAIN_Ration = 0.4;
N_TRAIN      = round(NB_PERSON * TRAIN_Ration);
N_TEST       = NB_PERSON - N_TRAIN;
dataTrain = cell(NB_CAMERA, 1); dataTest = cell(NB_CAMERA, 1);
for cam = 1:NB_CAMERA
    for pers = 1:N_TRAIN
        dataTrain{cam} = cat(3, dataTrain{cam}, dataSet{cam,pers});
    end
    for pers = N_TRAIN+1:NB_PERSON
        dataTest{cam} = cat(3, dataTest{cam}, dataSet{cam,pers});
    end
end

%% ----------------- Training Phase ----------------- %%
D = 30;
fisherAxes = train2DLDA(dataTrain, D);

d = 1:D;
COL  = 6;
LINE = round(length(d) / COL);
figure;

imageShow = dataTest{1}(:,:,2);
%subplot(1,length(d)+1,1); imshow(mat2gray(imageShow)); title('Original Image');
for i = 1:length(d)
    fisherFeature = imageShow * fisherAxes(:,1:d(i));
    
    % Reconstruction
    fisherReconstruction = fisherFeature * fisherAxes(:,1:d(i))';
    %subplot(1,length(d)+1,i+1); imshow(mat2gray(fisherReconstruction)); title(['d = ', num2str(d(i))]);
    subplot(LINE, COL, i); imshow(mat2gray(fisherReconstruction)); title(['d = ', num2str(d(i))]);
end

fisherFeatures = cell(NB_CAMERA,NB_PERSON);
for cam = 1:NB_CAMERA
    fisherFeatures{cam} = zeros(size(dataTest{cam},1), d(end));
    for i = 1:size(dataTest{cam},3)
        fisherFeatures{cam}(:,:,i) = dataTest{cam}(:,:,i) * fisherAxes(:,1:d(end));
    end
end


%% ----------------- Testing Phase ----------------- %%
%dataTrain   = dataSet(:,        1:N_TRAIN);
%dataTest    = dataSet(:,N_TRAIN+1:    end);

for i = 1:NB_CAMERA
    disp(size(dataTest{i}));
end
accuracy = zeros(NB_CAMERA, NB_CAMERA);
for cameraSource = 1:NB_CAMERA
    for cameraTarget = 1:NB_CAMERA
        for i = 1:size(dataTest{cameraSource,1},3)
            P_Source = dataTest{cameraSource}(:,:,i) * fisherAxes(:,d);
            P_Target = dataTest{cameraSource}(:,:,i) * fisherAxes(:,d);

            P_Bad    = cell(NB_CAMERA - 2, 1);
            for badCamera = 1:NB_CAMERA
                if badCamera == cameraSource || badCamera == cameraTarget
                    continue;
                end
                P_Bad{badCamera} = dataTest{cameraSource}(:,:,i) * fisherAxes(:,d);
            end

            %2/ Good Camera
            P_Good = dataTest{cameraTarget}(:,:,i) * fisherAxes(:,d);
            rightDist = norm(P_Good - P_Target);

            %3/ Bad Camera
            wrongDist = zeros(NB_CAMERA - 2,NB_PERSON - N_TRAIN,1);
            cnt_pass = 0;
            for badCamera = 1:NB_CAMERA
                if badCamera == cameraSource || badCamera == cameraTarget
                    cnt_pass = cnt_pass + 1;
                    continue;
                end
                for j = 1:NB_PERSON - N_TRAIN
                    P_Wrong = dataTest{badCamera}(:,:,j) * fisherAxes(:,d);
                    wrongDist(badCamera - cnt_pass,j) = norm(P_Wrong - P_Bad{badCamera});
                end
            end
            [minValue, minIdx]  = min(wrongDist);
            [minValue, ~]       = min(minValue);
            class  = (minValue < rightDist) * minIdx(1) + (minValue >= rightDist) * cameraTarget;
            %class = (minIdx < rightDist) * minIdx + (minIdx >= rightDist) * cameraTarget;
        
            accuracy(cameraSource,cameraTarget) = accuracy(cameraSource,cameraTarget) + (class == cameraTarget);
        end
    end
end

accuracy = accuracy / (NB_PERSON - N_TRAIN);
disp(accuracy);

figure("Name", "Ou est-ce que j'ai rangé Pull ?")
plotConfusionMatrix(accuracy, '2D-LDA');

Source = dataTest{1}(:,:,1) * fisherAxes(:,d); 
Target = dataTest{2}(:,:,1) * fisherAxes(:,d); dist_Target = norm(Target - Source);

Other  = cell(NB_PERSON - N_TRAIN);
for i = 1:NB_PERSON - N_TRAIN
    Other{i} = dataTest{3}(:,:,i) * fisherAxes(:,d);
end
figure("Name", "Ou est mon Pull putain ?")
subplot(NB_CAMERA, NB_PERSON - N_TRAIN, 1            ); imshow(mat2gray(Source)); title('Source');
subplot(NB_CAMERA, NB_PERSON - N_TRAIN, 1 + (NB_PERSON - N_TRAIN)); imshow(mat2gray(Target)); title(num2str(dist_Target));
for i = 1:NB_PERSON - N_TRAIN
    dist_Wrong = norm(Other{i} - Source);
    subplot(NB_CAMERA, NB_PERSON - N_TRAIN, i + ((NB_PERSON - N_TRAIN) * 2)); imshow(mat2gray(Other{i})); title(num2str(dist_Wrong));
end

