clear all; close all; 
addpath('./tools/');

DATASET_PATH = './Selected_Frames_Image';
load([DATASET_PATH, '/Selected_Frames.mat']);

%% Select N pertinenet frames
N = 5;
BEST_FRAME      = cell(NB_CAMERA, NB_PERSON);
BEST_MASK       = cell(NB_CAMERA, NB_PERSON);
UPGRADE_MASK    = cell(NB_CAMERA, NB_PERSON);
BEST_ROI        = cell(NB_CAMERA, NB_PERSON);        

for cam = 1:NB_CAMERA
    for pers = 1:NB_PERSON
        if isempty(SELECTED_FRAMES{cam,pers})
            continue;
        end
        BEST_ROI{cam, pers} = cell(N_SELECTED_FRAMES(cam,pers), 1);
        minSize = [Inf, Inf];
        for i = 1:N_SELECTED_FRAMES(cam,pers)
            stats = regionprops(SELECTED_MASK{cam,pers}(:,:,i), 'BoundingBox',  'Area');
            [~, idxMax] = max([stats.Area]);
            stats = stats(idxMax);
            BEST_ROI{cam,pers}{i} = SELECTED_FRAMES{cam,pers}(round(stats.BoundingBox(2)):floor(stats.BoundingBox(2)+stats.BoundingBox(4)), ...
                                                              round(stats.BoundingBox(1)):floor(stats.BoundingBox(1)+stats.BoundingBox(3)), i);
            BEST_FRAME{cam,pers}(:,:,i) = SELECTED_FRAMES{cam,pers}(:,:,i);
            
            minSize = min(minSize, size(BEST_ROI{cam,pers}{i}));
        end
    end
end

ROI_Resize = cell(NB_CAMERA, NB_PERSON);
for cam = 1:NB_CAMERA
    for pers = 1:NB_PERSON
        if isempty(BEST_ROI{cam,pers})
            continue;
        end
        for i = 1:N_SELECTED_FRAMES(cam,pers)
            ROI_Resize{cam,pers}(:,:,i) = mat2gray(imresize(BEST_ROI{cam,pers}{i}, minSize), [0, 1]);
            if((cam == 2 || cam == 3) && pers == 3)
                imwrite(ROI_Resize{cam,pers}(:,:,i), ['Publication/Processing/ROI_Camera_', num2str(cam), '_Person_', num2str(pers), '_Frame_', num2str(i), '.png']);
            end
        end
        
    end
end

% Display ROI_Resize
%for cam = 1:NB_CAMERA
%    figure("Name", ['Camera ', num2str(cam)]);
%    for pers = 1:NB_PERSON
%        if isempty(ROI_Resize{cam,pers})
%            continue;
%        end
%        for i = 1:N_SELECTED_FRAMES(cam,pers)
%            subplot(NB_PERSON, N_SELECTED_FRAMES(cam,pers), (pers-1)*N_SELECTED_FRAMES(cam,pers) + i);
%            imshow(mat2gray(ROI_Resize{cam,pers}(:,:,i)));
%        end
%    end
%end

%event = buildEvent(BEST_FRAME, N, '2D');
%
%for cam = 1:NB_CAMERA
%    figure("Name", ['Camera ', num2str(cam)]);
%    for pers = 1:NB_PERSON
%        if isempty(event{cam,pers})
%            continue;
%        end
%        for n = 1:N
%            subplot(NB_PERSON, N, (pers-1)*N + n);
%            imshow(mat2gray(event{cam,pers}(:,:,n)));
%        end
%    end
%end

%% Gaussian Model
gauss = @(x, mu, sigma) (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) ./ sigma).^2);
gauss = @(x, mu, sigma) exp(-0.5 * ((x - mu) ./ sigma).^2);
% 1/ Compute the mean and variance of the event
NB_CAMERA = 3;
event_mu            = cell(NB_CAMERA, NB_PERSON);
event_sigma     = cell(NB_CAMERA, NB_PERSON);
event_distribution  = cell(NB_CAMERA, NB_PERSON);
%x = 0:1/(prod(minSize)-1):1;
x = 0:0.01:1;
for cam = 1:NB_CAMERA
    for pers = 1:NB_PERSON  
        if isempty(ROI_Resize{cam,pers})
            continue;
        end
        
        data = zeros(prod(minSize), N_SELECTED_FRAMES(cam,pers));
        for i = 1:N_SELECTED_FRAMES(cam,pers)
            data(:,i) = reshape(ROI_Resize{cam,pers}(:,:,i), [], 1);
        end
        mu          = mean(data, 'all');
        variances   = std(data, 0, 'all');

        event_mu       {cam,pers} = mu;   
        event_sigma{cam,pers} = variances;

        %func = (@(x) (1 / (variances * 2 * pi)) * exp(-0.5 * ((x - mu) ./ variances).^2 ));
        event_distribution{cam, pers} = gauss(x, mu, variances)';
    end
    %event_distribution{cam, NB_PERSON + 1} = sum(cell2mat(event_distribution(cam,:)),2) / NB_PERSON;
    %event_mu{cam, NB_PERSON + 1} = mean(event_distribution{cam, NB_PERSON + 1});
    %event_sigma{cam, NB_PERSON + 1} = var(event_distribution{cam, NB_PERSON + 1});
end

cam = 1;
figure("Name", ['Camera ', num2str(cam)]);
for pers = 1:NB_PERSON
    hold on; plot(x,event_distribution{cam,pers}); hold off;
end
grid on; grid minor;
person = arrayfun(@(x) ['Person ', num2str(x)], 1:NB_PERSON, 'UniformOutput', false);
legend([person, 'Total']);

%X1 = cell2mat(event_distribution(1, 1));
%X2 = cell2mat(event_distribution(2, 1));
%
%X1_Tot = event_distribution{1, NB_PERSON + 1};
%X2_Tot = event_distribution{2, NB_PERSON + 1};
%
%subplot(2, 1, 1); plot(x, X1, x, X1_Tot); grid on; grid minor; legend('Person 1', 'Total'); title('Camera 1');
%subplot(2, 1, 2); plot(x, X2, x, X2_Tot); grid on; grid minor; legend('Person 1', 'Total'); title('Camera 2');
%
%M = X2_Tot * pinv(X1_Tot);
%predic = M * X1;
%
%subplot(2, 1, 2); hold on; plot(x,predic); hold off;
%subplot(2, 1, 2); hold on; plot(x,X1); hold off;

transformation_m = cell(NB_CAMERA, NB_CAMERA); transformation_mu    = cell(NB_CAMERA, NB_CAMERA);
transformation_v = cell(NB_CAMERA, NB_CAMERA); transforamtion_var   = cell(NB_CAMERA, NB_CAMERA);
M = cell(NB_CAMERA, NB_CAMERA); transformation_dist  = cell(NB_CAMERA, NB_CAMERA);
L = 1; %ceil(0.4 * NB_PERSON);
for camSource = 1:NB_CAMERA
    for camTarget = 1:NB_CAMERA

        source_MU   = cell2mat(event_mu(camSource, 1:L)           );
        target_MU   = cell2mat(event_mu(camTarget, 1:L)           );
        source_VAR  = cell2mat(event_sigma(camSource, 1:L)    );
        target_VAR  = cell2mat(event_sigma(camTarget, 1:L)    );
        source_D    = cell2mat(event_distribution(camSource, 1:L) );
        target_D    = cell2mat(event_distribution(camTarget, 1:L) );
        %[transformation_m{camSource, camTarget}, ~, transformation_mu{camSource, camTarget}]    = polyfit(source_MU , target_MU , 2);
        %[transformation_v{camSource, camTarget}, ~, transforamtion_var{camSource, camTarget}]   = polyfit(source_VAR, target_VAR, 2);
        %[M{camSource, camTarget}, ~, transformation_dist{camSource, camTarget}]  = polyfit(source_D  , target_D  , 2);
        transformation_m{camSource, camTarget} = target_MU  * pinv(source_MU);
        transformation_v{camSource, camTarget} = target_VAR * pinv(source_VAR);
        M{camSource, camTarget} = target_D   * pinv(source_D);
    end
end

% Compute the accuracy
%accuracyMean = confusionMatrix(event_mu          , transformation_m, 'poly', transformation_mu);
%accuracyVar   = confusionMatrix(event_sigma   , transformation_v, 'poly', transforamtion_var);
%accuracyDist   = confusionMatrix(event_distribution, M, 'poly', transformation_dist);
accuracyMean  = confusionMatrix(event_mu          , transformation_m);
accuracyVar   = confusionMatrix(event_sigma   , transformation_v);
accuracyDist  = confusionMatrix(event_distribution, M);
figure("Name", "Accuracy");
plotConfusionMatrix(accuracyMean, 'Mean Gaussian');


figure("Name", "Accuracy");
plotConfusionMatrix(accuracyVar, 'Variance Gaussian');


figure("Name", "Accuracy");
plotConfusionMatrix(accuracyDist, 'Distribution Gaussian');

% Display Distribution
figure("Name", "Distribution");
for cam = 1:NB_CAMERA
    for pers = 1:NB_PERSON
        if isempty(event_distribution{cam,pers})
            continue;
        end
        subplot(NB_CAMERA, 1, cam); 
        hold on; plot(x,event_distribution{cam,pers}); hold off;
        grid on; grid minor;
        title(['Camera ', num2str(cam)]);
    end
end

 % Plot transformation 2 to 1 
figure("Name", "Transformation Camera 2 to 1");
for pers = 1:NB_PERSON
    if isempty(event_distribution{1,pers}) || isempty(event_distribution{2,pers})
        continue;
    end

    %        mean_prediction = polyval(transformation_m{2,1}, event_mu{2,pers}       , [], transformation_mu{2,1});
    %         var_prediction = polyval(transformation_v{2,1}, event_sigma{2,pers}, [], transforamtion_var{2,1});
    %distribution_prediction = polyval(M{2,1}, event_distribution{2,pers}, [], transformation_dist{2,1});
    mean_prediction = transformation_m{2,1} * event_mu{2,pers};
    var_prediction  = transformation_v{2,1} * event_sigma{2,pers};
    distribution_prediction = M{2,1} * event_distribution{2,pers};

    prediction = gauss(x, mean_prediction, var_prediction);

    subplot(NB_PERSON, 1, pers);
    hold on; plot(x, prediction, '*'); 
             plot(x, distribution_prediction);
             plot(x, event_distribution{1,pers}); 
             hold off;
    grid on; grid minor;
    legend('Prediction using Mean and Variance', 'Prediction using Distribution', 'Real');
    title(['Person ', num2str(pers)]);

end

% Plot parameters of transformation 2 to 1
for pers = 1:NB_PERSON
    if isempty(event_distribution{1,pers}) || isempty(event_distribution{2,pers})
        continue;
    end
    %mean_prediction = polyval(transformation_m{2,1}, event_mu{2,pers}       , [], transformation_mu{2,1});
    %var_prediction  = polyval(transformation_v{2,1}, event_sigma{2,pers}, [], transforamtion_var{2,1});

    mean_prediction = transformation_m{2,1} * event_mu{2,pers};
    var_prediction  = transformation_v{2,1} * event_sigma{2,pers};

    % Print the parameters
    fprintf('Prediction of Person %d: Mean: %f, Variance: %f\n', pers, mean_prediction, var_prediction);
    fprintf('Real       of Person %d: Mean: %f, Variance: %f\n', pers, event_mu{1,pers}, event_sigma{1,pers});

end

for cam = 2:3
    for pers = 3:3
        if isempty(event_distribution{cam,pers})
            continue;
        end
        figure("Name", ['Camera ', num2str(cam), ' Person ', num2str(pers)]);
        hold on; plot(x,event_distribution{cam,pers}); hold off;
        grid on; grid minor;
        title('Gaussian Distribution');
        saveas(gcf, ['Publication/Processing/Gaussian_Camera_', num2str(cam), '_Person_', num2str(pers), '.png']);

    end
end
prediction = M{2,3} * event_distribution{2,3};
save('Publication/Processing/ploting.mat', 'event_distribution', 'prediction', 'x', 'ROI_Resize', 'BEST_FRAME');



camSource = 1; camTarget = 2;
figure("Name", "Prediction");
for pers = 1:NB_PERSON
    prediction = M{camSource, camTarget} * event_distribution{camSource, pers};
    subplot(NB_PERSON, 1, pers);
    hold on; plot(x, prediction); hold off;
    hold on; plot(x, event_distribution{camTarget, pers}); hold off;
    grid on; grid minor;
    legend('Prediction', 'Real');
    title(['Person ', num2str(pers)]);
end

%% Training Test
M = cell(NB_CAMERA, NB_CAMERA)
figure("Name", "Training Evolution");
camSource = 1; camTarget = 2;
for pers = 1:NB_PERSON
    if isempty(M{camSource, camTarget})
        M{camSource, camTarget} = event_distribution{camTarget, pers} * pinv(event_distribution{camSource, pers});    
        mu_Source      = event_mu{camSource, pers};         mu_Target      = event_mu{camTarget, pers};
        sigma_Source   = event_sigma{camSource, pers};      sigma_Target   = event_sigma{camTarget, pers};
    else
        mu_Source    = ((mu_Source    * pers) + event_mu{camSource, pers} / pers);
        sigma_Source = ((sigma_Source * pers) + event_sigma{camSource, pers} / pers);

        mu_Target    = ((mu_Target    * pers) + event_mu{camTarget, pers} / pers);
        sigma_Target = ((sigma_Target * pers) + event_sigma{camTarget, pers} / pers);

        noise_Source = gauss(x, mu_Source, sigma_Source)';
        noise_Target = gauss(x, mu_Target, sigma_Target)';
        
        source = [cell2mat(event_distribution(camSource, pers)), noise_Source];
        target = [cell2mat(event_distribution(camTarget, pers)), noise_Target];

        M{camSource, camTarget} = target * pinv(source);
    end
    subplot(1,NB_PERSON,pers);
    hold on;
    predict = M{camSource, camTarget} * event_distribution{camSource, pers};
    plot(x, predict);
    plot(x, event_distribution{camTarget, pers});
    hold off;
    grid on; grid minor;
    legend('Prediction', 'Real');
    title(['Person ', num2str(pers)]);
end

% Test the prediction
figure("Name", "Prediction");
for pers = 1:NB_PERSON
    prediction = M{camSource, camTarget} * event_distribution{camSource, pers};
    subplot(NB_PERSON, 1, pers);
    hold on; plot(x, prediction); hold off;
    hold on; plot(x, event_distribution{camTarget, pers}); hold off;
    grid on; grid minor;
    legend('Prediction', 'Real');
    title(['Person ', num2str(pers)]);
end
%%
 



% Curve of Person 3 in Camera 2 and 3

for cam = 2:3
    for pers = 3:3
        if isempty(event_distribution{cam,pers})
            continue;
        end
        figure("Name", ['Camera ', num2str(cam), ' Person ', num2str(pers)]);
        hold on; plot(x,event_distribution{cam,pers}); hold off;
        grid on; grid minor;
        title('Gaussian Distribution');
        saveas(gcf, ['Publication/Processing/Gaussian_Camera_', num2str(cam), '_Person_', num2str(pers), '.png']);

    end
end
prediction = M{2,3} * event_distribution{2,3};
save('Publication/Processing/ploting.mat', 'event_distribution', 'prediction', 'x');

figure("Name", "Accuracy");
hold on; plot(x, prediction); hold off;
hold on; plot(x, event_distribution{3,3}); hold off;
grid on; grid minor;
legend('Prediction', 'Real');
title('Gaussian Distribution');
saveas(gcf, 'Publication/Processing/Gaussian_Prediction.png');




%accuracy = zeros(NB_CAMERA, NB_CAMERA);
%for cameraSource = 1:NB_CAMERA
%    for cameraTarget = 1:NB_CAMERA
%        cnt = 0;
%        for person = 1:NB_PERSON
%            if isempty(event_distribution{cameraSource,person}) || isempty(event_distribution{cameraTarget,person})
%                continue;
%            end
%            cnt = cnt + 1;
%            % 1/ Init Camera
%            P_Source = event_distribution{cameraSource,person};            
%            %2/ Good Camera
%            P_Target = event_distribution{cameraTarget,person};
%            rightDist = mse(P_Target, P_Source);
%
%            %3/ Bad Camera
%            wrongDist = zeros(NB_CAMERA,NB_PERSON,1) * inf;
%            for badCamera = 1:NB_CAMERA
%                if (badCamera == cameraSource || badCamera == cameraTarget) || isempty(event_distribution{badCamera})
%                    continue;
%                end
%                for i = 1:NB_PERSON
%                    if i == person || isempty(event_distribution{badCamera,i})
%                        continue;
%                    end
%                    P_Wrong   = event_distribution{badCamera,i};
%                    wrongDist(badCamera,i) = mse(P_Wrong, P_Source);
%                end
%            end
%            [minValue, minIdx]  = min(wrongDist);
%            [minValue, ~]       = min(minValue);
%            class  = (minValue < rightDist) * minIdx(1) + (minValue >= rightDist) * cameraTarget;
%            accuracy(cameraSource,cameraTarget) = accuracy(cameraSource,cameraTarget) + (class == cameraTarget);
%        end
%        accuracy(cameraSource,cameraTarget) = accuracy(cameraSource,cameraTarget) / cnt;
%    end
%end
%
%figure("Name", "Accuracy");
%plotConfusionMatrix(accuracy, 'Gaussian');