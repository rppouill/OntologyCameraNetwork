clear all; close all; % clc;
addpath('./tools'); addpath('./Transformation'); 

%DATASET_NAME = 'ImageSelectedwithD/';
DATASET_NAME = 'ImageSelected/';
%DATASET_NAME = 'NewDataset/';

RESULT_FOLDER = ['results/', DATASET_NAME];
load([RESULT_FOLDER, 'transformations.mat']);

N_VECTOR_EXAMPLE = 5;
nVector = N_VECTOR_EXAMPLE;



%% Credit with Disp %%
disp('Medium Processing');
disp('Author: Romain POUILLARD');
disp('Date: 2025-01-05');
disp('Version: 1.0');


Global_M = {M_LS, M_LSA, M_GR, M_PL};
Global_Method = {'Least Square', 'Least Square Adaptatif', 'Gradient Descent', 'Poly'};

figure("Name", "Confusion Matrix");
for meth = 1:size(Global_M,2)
    M = Global_M{meth};
    Method = Global_Method{meth};
        
    set(gcf, 'Position', [100, 100, 1200, 400]);
    txt = 'Yaousa';
    if meth == 4
        txt = 'poly';
    end
        
    accuracy_Test = confusionMatrix(eigVec(:,L+1:end), M, txt);
    subplot(1,4,meth); plotConfusionMatrix(accuracy_Test, 'Dataset Test');
    title(Method);
    
    %saveas(gcf, [RESULT_FOLDER, Method, '.png']);

end



%accuracy_LS     = confusionMatrix(eigVec, M_LS);
%accuracy_LSA    = confusionMatrix(eigVec, M_LSA);
%accuracy_GR     = confusionMatrix(eigVec, M_GR);
%
%
%
%
%plotConfusionMatrix(accuracy_LS, 'LS');
%plotConfusionMatrix(accuracy_LSA, 'LSA');
%plotConfusionMatrix(accuracy_GR, 'GR');

%accuracy_LS_Train   = confusionMatrix(eigVec(:,1:L), M_LS);
%accuracy_LSA_Train  = confusionMatrix(eigVec(:,1:L), M_LSA);
%accuracy_GR_Train   = confusionMatrix(eigVec(:,1:L), M_GR);
%
%plotConfusionMatrix(accuracy_LS_Train, 'LS Train');
%plotConfusionMatrix(accuracy_LSA_Train, 'LSA Train');
%plotConfusionMatrix(accuracy_GR_Train, 'GR Train');
%
%
%accuracy_LS_Test    = confusionMatrix(eigVec(:,L+1:end), M_LS);
%accuracy_LSA_Test   = confusionMatrix(eigVec(:,L+1:end), M_LSA);
%accuracy_GR_Test    = confusionMatrix(eigVec(:,L+1:end), M_GR);
%
%plotConfusionMatrix(accuracy_LS_Test, 'LS Test');
%plotConfusionMatrix(accuracy_LSA_Test, 'LSA Test');
%plotConfusionMatrix(accuracy_GR_Test, 'GR Test');
