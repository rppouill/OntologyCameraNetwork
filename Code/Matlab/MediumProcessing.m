clear all; close all; clc;
addpath('./tools');

DATASET_PATH = './dataSet/';
NB_CAMERA = 2; NB_PERSON = 9; 
NB_VECTOR = 8; IMG_SIZE  = [30,30];


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

%% Compute the Projection between Camera 1 and 2 without transformation %%
proj = projection(pcaEigVec(1,:),pcaEigVec(2,:));
save('debugFile/projection_new.mat','proj');
[oui,non] = projectionCompare(proj);
disp(['oui = ',num2str(oui),', non = ',num2str(non)]);


%% First Transformation %%
% Spatial
M = pcaEigVec{2,1} * pinv(pcaEigVec{1,1});

W_T = cell(NB_PERSON,1);
for i = 1:NB_PERSON
    W_T{i} = M * pcaEigVec{1,i};
end

projM = projection(W_T,pcaEigVec(2,:));
[ouiM,nonM] = projectionCompare(projM);
disp(['oui = ',num2str(ouiM),', non = ',num2str(nonM)]);

%% Second Transformation %%
% 3 Magic Matrix

M = pcaEigVec{2,1} * pinv(pcaEigVec{1,1});
A = cell(NB_PERSON,1); W_3M = cell(NB_PERSON,1);
for i = 1:NB_PERSON
    X_1 = pcaEigVec{1,i} * pinv(pcaEigVec{1,1});
    X_2 = pcaEigVec{2,i} * pinv(pcaEigVec{2,1});
    C = X_2 * M * pinv(X_1);
    A{i} = X_2 * M * pinv(X_1) * pinv(X_1);

    W_3M{i} = A{i} * pcaEigVec{1,i};
end

proj3M = projection(W_3M,pcaEigVec(2,:));
[oui3M,non3M] = projectionCompare(proj3M);
disp(['oui = ',num2str(oui3M),', non = ',num2str(non3M)]);


%% Third Transformation %%
% Total Projection
M = pcaEigVec{2,1} * pinv(pcaEigVec{1,1});
B = cell(NB_PERSON,NB_PERSON); W_TM = cell(NB_PERSON, NB_PERSON);

for i = 1:NB_PERSON
    X_1 = pcaEigVec{1,i} * pinv(pcaEigVec{1,1});
    for j = 1:NB_PERSON
        X_2 = pcaEigVec{2,j} * pinv(pcaEigVec{2,1});
        B{i,j} = X_1 * M * pinv(X_2);
        W_TM{i,j} = B{i,j} * pcaEigVec{1,j};
    end
end

projTM = projection(W_TM,pcaEigVec(2,:));
[ouiTM,nonTM] = projectionCompare(projTM);
disp(['oui = ',num2str(ouiTM),', non = ',num2str(nonTM)]);


%% Fourth Transformation %%
% Single Transformation
M = pcaEigVec{2,1} * pinv(pcaEigVec{1,1});
A = cell(NB_PERSON,1); W_SM = cell(NB_PERSON,1);
for i = 1:NB_PERSON
    X = pcaEigVec{1,i} * pinv(pcaEigVec{1,1});
    A{i} = X * M;
    W_SM{i} = A{i} * pcaEigVec{1,i};
end

projSM = projection(W_SM,pcaEigVec(2,:));
[ouiSM,nonSM] = projectionCompare(projSM);
disp(['oui = ',num2str(ouiSM),', non = ',num2str(nonSM)]);