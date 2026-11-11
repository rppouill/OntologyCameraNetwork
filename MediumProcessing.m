clear all; close all; % clc;
addpath('./tools');

DATASET_PATH = './dataSet/';
DATASET_NAME = 'ImageSelected/';
NB_CAMERA = 3; NB_PERSON = 9; 
NB_VECTOR = 8; IMG_SIZE  = [30,30];

N_VECTOR_EXAMPLE = 5; SHOW = false;


%% Credit with Disp %%
disp('Medium Processing');
disp('Author: Romain POUILLARD');
disp('Date: 2021-06-01');
disp('Version: 1.0');

%if isempty(gcp('nocreate'))
%    parpool;
%end 
%% Load the image of Camera %%
image_selected = [DATASET_PATH, DATASET_NAME];

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

if exist([image_selected, 'D.mat'], 'file')
    load([image_selected, 'D.mat']);
    D = cellfun(@(x) normalize(x, 'range'), D, 'UniformOutput', false);
else
    D = cell(NB_CAMERA, NB_PERSON);
    for i = 1:NB_CAMERA
        for j = 1:NB_PERSON
            D{i,j} = ones(1, size(camera{i,j},2));
        end
    end
end
eigVec = buildEvent(camera, NB_VECTOR, D);

% Result directly 
RESULT_FOLDER = ['results/', DATASET_NAME];
if ~exist(RESULT_FOLDER, 'dir')
    mkdir(RESULT_FOLDER);
end

accuracyMetric = zeros(1, NB_VECTOR);
for nVector = 1:NB_VECTOR
    pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false); 
    for i = 1:NB_CAMERA
        for j = 1:NB_CAMERA
            %if i == j
            %    continue;
            %end
            I = eye(IMG_SIZE(1) * IMG_SIZE(2), IMG_SIZE(1) * IMG_SIZE(2));
            accuracyMetric(nVector) = accuracyMetric(nVector) + evaluate(pcaEigVec(i,:), pcaEigVec(j,:), I);
        end
    end
    accuracyMetric(nVector) = accuracyMetric(nVector) / (NB_CAMERA * NB_CAMERA);
end

figure("Name", "Plot Dataset")
pcaEigVec = cellfun(@(x) x(:,1:1), eigVec, 'UniformOutput', false);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        subplot(NB_CAMERA,NB_PERSON,j + (i-1) * NB_PERSON);
        imshow(mat2gray(reshape(pcaEigVec{i,j}(:,1),IMG_SIZE)));
    end
end
saveas(gcf, [RESULT_FOLDER, 'Dataset.png']);

%% Dataset for the Training %%
training_ratio = 0.7;
L = floor(NB_PERSON * training_ratio);

%% ---------- Least Square Method ---------- %%
tic 
range_nVector = 1:NB_VECTOR;
accuracyMetricLS = zeros(1, length(range_nVector));
M_LS = cell(NB_CAMERA, NB_CAMERA, NB_VECTOR);
for nVector = range_nVector
    %% Compute the Eigentarget %%
    pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false); 
    %% Main Processing %%
    for i = 1:NB_CAMERA
        for j = 1:NB_CAMERA
            %if i == j
            %    continue;
            %end

            A = cell2mat(pcaEigVec(i,1:L));
            B = cell2mat(pcaEigVec(j,1:L));

            M = polyfit(A, B, 1);
            X    = M(1);
            bias = M(2);

            M_LS{i,j,nVector} = X;
            accuracyMetricLS(nVector) = accuracyMetricLS(nVector) + evaluate(pcaEigVec(i,L:end), pcaEigVec(j,L:end), X, bias);

            if nVector == N_VECTOR_EXAMPLE && i == 1 && j == 2 && SHOW
                evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X, true);
                saveas(gcf, [RESULT_FOLDER, 'LeastSquare.png']);
            end
        end
    end
    accuracyMetricLS(nVector) = accuracyMetricLS(nVector) / (NB_CAMERA * NB_CAMERA);

end
toc

%% ---------- Least Square Adaptative Method ---------- %% 
% For each new person we adapt the transformation matrix using "Hebb Rules"

tic 
accuracyMetricLSA = zeros(1, length(range_nVector));
M_LSA = cell(NB_CAMERA, NB_CAMERA, NB_VECTOR);

for nVector = range_nVector
    %% Compute the Eigentarget %%
    pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false);
    
    %% Main Processing %%
    alpha = 0.7;
    for i = 1:NB_CAMERA
        for j = 1:NB_CAMERA
            %if i == j
            %    continue;
            %end
            
            M = polyfit(pcaEigVec{j,1}, pcaEigVec{i,1}, 1);
            X = M(1); bias = M(2);
            for k = 2:L
                M = polyfit(pcaEigVec{j,k}, pcaEigVec{i,k}, 1);
                X = (X * alpha) + (M(1) * (1 - alpha));
                bias = (bias * alpha) + (M(2) * (1 - alpha));
            end
            M_LSA{i,j,nVector} = X;
            
            if nVector == N_VECTOR_EXAMPLE && i == 1 && j == 2 && SHOW
                evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X, true);
                saveas(gcf, [RESULT_FOLDER, 'LeastSquareAdaptative.png']);
            end

            accuracyMetricLSA(nVector) = accuracyMetricLSA(nVector) + evaluate(pcaEigVec(i,L:end), pcaEigVec(j,L:end), X, bias);
        end
    end
    
    accuracyMetricLSA(nVector) = accuracyMetricLSA(nVector) / (NB_CAMERA * NB_CAMERA);

end

toc
%% Find Best couple alpha - vector with Least Square Adaptative %%
alpha = 0.1:0.1:1;
alphaMetricLSA = zeros(nVector, length(alpha));
for nVector = range_nVector
    for a = alpha
        pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false);

        for i = 1:NB_CAMERA
            for j = 1:NB_CAMERA
                %if i == j
                %    continue;
                %end

                X = pcaEigVec{j,1} * pinv(pcaEigVec{i,1});
                for k = 2:L
                    X_tmp = pcaEigVec{j,k} * pinv(pcaEigVec{i,k});
                    X = (X * a) + (X_tmp * (1 - a));
                end
                alphaMetricLSA(nVector, round(a*10)) = mean([evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X) alphaMetricLSA(nVector, round(a*10))]);
            end
        end
    end
end

figure("Name", "Alpha Metric");
contourf(alpha, range_nVector, alphaMetricLSA);
colorbar; colormap('jet'); grid on; grid minor;
xlabel('Alpha');
ylabel('Number of Vector'); 
zlabel('Accuracy');         zlim([0, 1]);
title('Accuracy evolution with Alpha and Number of Vector');
saveas(gcf, [RESULT_FOLDER, 'AlphaMetric.png']);


%% ---------- Gradient Descent Method ---------- %%
tic
accuracyMetricGR = zeros(1, length(range_nVector));
M_GR = cell(NB_CAMERA, NB_CAMERA,NB_VECTOR);
for nVector = range_nVector
    %% Compute the Eigentarget %%
    pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false);
    
    %% Main Processing %%
    eta = 0.01;
    for i = 1:NB_CAMERA
        for j = 1:NB_CAMERA
            %if i == j
            %    continue;
            %end

            X       = pcaEigVec{j,1} * pinv(pcaEigVec{i,1});
            bias    = (2/NB_CAMERA) * sum(pcaEigVec{j,1} - X * pcaEigVec{i,1});
            for k = 1:L
                pred    = X     * pcaEigVec{i,k} + bias;
                mse     = mean((pred - pcaEigVec{j,k}).^2);
                
                error   = pred - pcaEigVec{j,k};
                dw      = (2/NB_CAMERA) * (error * pcaEigVec{i,k}');
                db      = (2/NB_CAMERA) * sum(error);

                X       = X - eta * dw;
            end
            
            M_GR{i,j,nVector} = X;
            if nVector == N_VECTOR_EXAMPLE && i == 1 && j == 2 && SHOW
                evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X, true);
                saveas(gcf, [RESULT_FOLDER, 'GradientDescent.png']);
            end

            accuracyMetricGR(nVector) = accuracyMetricGR(nVector) + evaluate(pcaEigVec(i,L:end), pcaEigVec(j,L:end), X);
        end
    end
    accuracyMetricGR(nVector) = accuracyMetricGR(nVector) / (NB_CAMERA * NB_CAMERA);
end
toc

%% Find best tuple eta - vector with Gradient Descent %%
eta = 0.1:0.1:1;
etaMetricGR = zeros(nVector, length(eta));
for nVector = range_nVector
    for e = eta
        pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false);

        for i = 1:NB_CAMERA
            for j = 1:NB_CAMERA
                %if i == j
                %    continue;
                %end

                X = pcaEigVec{j,1} * pinv(pcaEigVec{i,1});
                for k = 2:L
                    pred    = X * pcaEigVec{i,k};
                    error   = pred - pcaEigVec{j,k};
                    grad    = error * pcaEigVec{i,k}';
                    X       = X - e * grad;
                end
                etaMetricGR(nVector, round(e*10)) = mean([evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X) etaMetricGR(nVector, round(e*10))]);
                %etaMetricGR(nVector, round(e*100)) = ... 
                %        mean([evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X) 
                %                etaMetricGR(nVector, round(e*100))]);
            
            end
        end
    end
end

figure("Name", "Eta Metric");
contourf(eta, range_nVector, etaMetricGR);
colorbar; colormap('jet'); grid on; grid minor;
xlabel('Eta'); 
ylabel('Number of Vector'); 
zlabel('Accuracy');         zlim([0, 1]);
title('Accuracy evolution with Eta and Number of Vector');
saveas(gcf, [RESULT_FOLDER, 'EtaMetric.png']);

%% Polynomial Regression %%
tic
accuracyMetricPL = zeros(1, length(range_nVector));
M_PL = cell(NB_CAMERA, NB_CAMERA, NB_VECTOR);
for nVector = range_nVector
    %% Compute the Eigentarget %%
    pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false);
    
    %% Main Processing %%
    order = 2;
    for i = 1:NB_CAMERA
        for j = 1:NB_CAMERA
            %if i == j
            %    continue;
            %end

            X = polyfit(pcaEigVec{i,1}, pcaEigVec{j,1}, order);
            M_PL{i,j,nVector} = X;
    
            if nVector == N_VECTOR_EXAMPLE && i == 1 && j == 2 && SHOW
                evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X, true);
                saveas(gcf, [RESULT_FOLDER, 'GradientDescent.png']);
            end

            accuracyMetricPL(nVector) = accuracyMetricPL(nVector) + evaluate(pcaEigVec(i,L:end), pcaEigVec(j,L:end), X, 'poly');
        end
    end
    accuracyMetricPL(nVector) = accuracyMetricPL(nVector) / (NB_CAMERA * NB_CAMERA);
end
toc
%% Find best tuple order - vector with Polynomial Regression %%
order = 1:9;
orderMetricPL = zeros(nVector, length(order));
for nVector = range_nVector
    for o = order
        pcaEigVec = cellfun(@(x) x(:,1:nVector), eigVec, 'UniformOutput', false);

        for i = 1:NB_CAMERA
            for j = 1:NB_CAMERA
                X = polyfit(pcaEigVec{i,1}, pcaEigVec{j,1}, o);
                orderMetricPL(nVector, o) = mean([evaluate(pcaEigVec(i,:), pcaEigVec(j,:), X, 'poly') orderMetricPL(nVector, o)]);
            end
        end
    end
end

figure("Name", "Order Metric");
contourf(order, range_nVector, orderMetricPL);
colorbar; colormap('jet'); grid on; grid minor;
xlabel('Order');
ylabel('Number of Vector');
zlabel('Accuracy');         zlim([0, 1]);
title('Accuracy evolution with Order and Number of Vector');
saveas(gcf, [RESULT_FOLDER, 'OrderMetric.png']);




figure("Name", "Accuracy");
hold on;
plot(range_nVector, accuracyMetric, '*-');
plot(range_nVector, accuracyMetricLS, '*-');
plot(range_nVector, accuracyMetricLSA,'*-');
plot(range_nVector, accuracyMetricGR, '*-');
plot(range_nVector, accuracyMetricPL, '*-');
hold off;

grid on; grid minor;
xlabel('Number of Eigenvector'); xlim([1, NB_VECTOR]);
ylabel('Accuracy');         ylim([0, 1]);
title('Accuracy along the number of Eigenvector');
legend('Without Transformation', 'Least Square', 'Least Square Adaptative', 'Gradient Descent');
saveas(gcf, [RESULT_FOLDER, 'Accuracy.png']);

% Save differents transformations matrix
save([RESULT_FOLDER, 'transformations.mat'], 'M_LS', 'M_LSA', 'M_GR','M_PL', 'eigVec', 'L', 'NB_VECTOR', 'NB_CAMERA', 'NB_PERSON', 'IMG_SIZE');


