function [] = trainCentroid(varargin)
    close all;
    addpath('./tools/');
    DATASET_PATH = './Selected_Frames_Image/';
    N            = 5;

    if mod(nargin, 2) == 1
        DATASET_PATH = varargin{1};
    end

    for i = 1:nargin
        if strcmp(varargin{i}, 'N')
            N = varargin{i+1};
        end
    end

    load([DATASET_PATH, '/Selected_Frames.mat']);

    % Select Pertiennet Frames
    BEST_FRAME = cell(NB_CAMERA, NB_PERSON);
    BEST_MASK   = cell(NB_CAMERA, NB_PERSON);
    UPGRADE_MASK = cell(NB_CAMERA, NB_PERSON);
    for cam = 1:NB_CAMERA
        for pers = 1:NB_PERSON
            if isempty(SELECTED_FRAMES{cam,pers})
                continue;
            end
            diff_D = abs(D{cam,pers} - MEAN_D(cam,pers));
            [~, idx] = sort(diff_D);
            idx = idx(1:N); idx = sort(idx);
            for i = 1:N
                BEST_FRAME{cam,pers}(:,:,i) = SELECTED_FRAMES{cam,pers}(:,:,idx(i));
                stats = regionprops(SELECTED_MASK{cam,pers}(:,:,idx(i)), 'Area', 'BoundingBox', 'ConvexImage');
                [~, idxMax] = max([stats.Area]);
                stats = stats(idxMax);
                buffMask = zeros(size(SELECTED_MASK{cam,pers}(:,:,idx(i))));
                buffMask(round(stats.BoundingBox(2)):floor(stats.BoundingBox(2)+stats.BoundingBox(4)), ...
                        round(stats.BoundingBox(1)):floor(stats.BoundingBox(1)+stats.BoundingBox(3))) = stats.ConvexImage;
                BEST_MASK  {cam,pers}(:,:,i) = SELECTED_MASK{cam,pers}(:,:,idx(i));
                UPGRADE_MASK{cam,pers}(:,:,i) = buffMask;

                WEIGTHED_FRAME{cam,pers}(:,:,i) = SELECTED_FRAMES{cam,pers}(:,:,idx(i)) .* UPGRADE_MASK{cam,pers}(:,:,i);

            end
        end
    end

    % Some Characteristics
    AREA      = cell(NB_CAMERA, 1);
    MEAN      = cell(NB_CAMERA, 1);
    PERIMETER = cell(NB_CAMERA, 1);

    for cam = 1:NB_CAMERA
        for pers = 1:NB_PERSON
            if isempty(SELECTED_FRAMES{cam,pers})
                continue;
            end

            for i = 1:N
                stats = regionprops(UPGRADE_MASK{cam,pers}(:,:,i), 'Area', 'Perimeter');
                subImage = UPGRADE_MASK{cam,pers}(:,:,i) .* BEST_FRAME{cam,pers}(:,:,i);
                PERIMETER{cam}(pers,i) = stats.Perimeter;
                     AREA{cam}(pers,i) = stats.Area;
                     MEAN{cam}(pers,i) = sum(subImage(:)) / stats.Area;

            end
        end
    end

    % Centroid of camera 1
    cam = 1;
    figure("Name", ['Centroid evolution of camera ', int2str(cam)]);

    plot3(     AREA{cam}(:,1), ...
          PERIMETER{cam}(:,1), ...
               MEAN{cam}(:,1), 'b*');
    text(     AREA{cam}(:,1), ...
          PERIMETER{cam}(:,1), ...
               MEAN{cam}(:,1), string(1:NB_PERSON));


    centroid = [AREA{cam}(1,1), PERIMETER{cam}(1,1), MEAN{cam}(1,1)];
    nSample = 1;
    for pers = 1:NB_PERSON
        if isempty(SELECTED_FRAMES{cam,pers})
            continue;
        end

        start = (pers == 1) + 1;
        for nFrame = start:N
            nSample = nSample + 1;
            centroid(1) = ((centroid(1) * (nSample-1)) +      AREA{cam}(pers,nFrame)) / nSample;
            centroid(2) = ((centroid(2) * (nSample-1)) + PERIMETER{cam}(pers,nFrame)) / nSample;
            centroid(3) = ((centroid(3) * (nSample-1)) +      MEAN{cam}(pers,nFrame)) / nSample;
        end
        hold on;
        plot3(centroid(1), centroid(2), centroid(3), 'r*');
        text(centroid(1), centroid(2), centroid(3), ['P', int2str(pers)]);
        hold off;
    end
    grid on; grid minor;
    xlabel('Area'); ylabel('Perimeter'); zlabel('Mean');
    title('Centroid evolution of camera 1');


    % Centroid
    TRAIN_RATIO = 0.7; 
    L = floor(NB_PERSON * TRAIN_RATIO);
    CENTROID = zeros(NB_CAMERA, 3);
    for cam = 1:NB_CAMERA
        CENTROID(cam,1) = mean(     AREA{cam}(:));
        CENTROID(cam,2) = mean(PERIMETER{cam}(:));
        CENTROID(cam,3) = mean(     MEAN{cam}(:));
    end


    % Train
         AREA_TRANSFORMATION = cell(NB_CAMERA, NB_CAMERA); AREA_S = cell(NB_CAMERA, NB_CAMERA); AREA_U = cell(NB_CAMERA, NB_CAMERA);
    PERIMETER_TRANSFORMATION = cell(NB_CAMERA, NB_CAMERA);
         MEAN_TRANSFORMATION = cell(NB_CAMERA, NB_CAMERA);

    for camSource = 1:NB_CAMERA
        for camDest = 1:NB_CAMERA
                 [AREA_TRANSFORMATION{camSource, camDest}, ...
                  AREA_S{camSource, camDest}, ...
                  AREA_U{camSource, camDest}]            = polyfit(     AREA{camSource}(1:L,:),      AREA{camDest}(1:L,:), 4);
            PERIMETER_TRANSFORMATION{camSource, camDest} = polyfit(PERIMETER{camSource}(1:L,:), PERIMETER{camDest}(1:L,:), 4);
                 MEAN_TRANSFORMATION{camSource, camDest} = polyfit(     MEAN{camSource}(1:L,:),      MEAN{camDest}(1:L,:), 4);
        end
    end

    % Display 1 to 2 transformation
    camSource = 1; camTarget = 2;
    figure("Name", ['Transformation from camera ', int2str(camSource), ' to camera ', int2str(camTarget)]);
    
    
    
    hold on;
    plot3(AREA{camSource}, PERIMETER{camSource}, MEAN{camSource}, 'g*');
    plot3(AREA{camTarget}, PERIMETER{camTarget}, MEAN{camTarget}, 'y*');

    plot3(CENTROID(:,1), CENTROID(:,2), CENTROID(:,3), 'ro');
     text(CENTROID(:,1), CENTROID(:,2), CENTROID(:,3), string(1:NB_CAMERA));
    hold off;
    
    for cam = [camSource,camTarget]
        for pers = 1:NB_PERSON
            hold on;
            plot3([CENTROID(cam,1), AREA{cam}(pers,1)], ...
                  [CENTROID(cam,2), PERIMETER{cam}(pers,1)],...
                  [CENTROID(cam,3), MEAN{cam}(pers,1)], 'r');
            hold off;
        end
    end
    
    for pers = 1:NB_PERSON
        if isempty(SELECTED_FRAMES{camSource,pers})
            continue;
        end
        predict(1,:) = polyval(     AREA_TRANSFORMATION{camSource, camTarget},      AREA{camSource}(pers,:), AREA_S{camSource, camTarget}, AREA_U{camSource, camTarget});
        predict(2,:) = polyval(PERIMETER_TRANSFORMATION{camSource, camTarget}, PERIMETER{camSource}(pers,:));
        predict(3,:) = polyval(     MEAN_TRANSFORMATION{camSource, camTarget},      MEAN{camSource}(pers,:));

        hold on;
        plot3(predict(1,:), predict(2,:), predict(3,:), 'b*');
         text(predict(1,:), predict(2,:), predict(3,:), num2str(pers));
        hold off;
    end
    legend('Camera 1', 'Camera 2', 'Centroid');
    grid on; grid minor;
    xlabel('Area'); ylabel('Perimeter'); zlabel('Mean');
    title(['Transformation from camera ', int2str(camSource), ' to camera ', int2str(camTarget)]);


    % Evale
    accuracy = zeros(NB_CAMERA, NB_CAMERA);
    for camSource = 1:NB_CAMERA
        for camTarget = 1:NB_CAMERA
            cnt = 0;
            for pers = 1:NB_PERSON
                if isempty(AREA{camSource}(pers,:)) || isempty(AREA{camTarget}(pers,:))
                    continue;
                end
                cnt = cnt + 1;
                % Source to Target
                predict(1,:) = polyval(     AREA_TRANSFORMATION{camSource, camTarget},      AREA{camSource}(pers,:), AREA_S{camSource, camTarget}, AREA_U{camSource, camTarget});
                predict(2,:) = polyval(PERIMETER_TRANSFORMATION{camSource, camTarget}, PERIMETER{camSource}(pers,:));
                predict(3,:) = polyval(     MEAN_TRANSFORMATION{camSource, camTarget},      MEAN{camSource}(pers,:));

                distance = min(sqrt(sum((predict - [      AREA{camTarget}(pers, 1); 
                                                     PERIMETER{camTarget}(pers, 1); 
                                                          MEAN{camTarget}(pers, 1)]).^2, 1)));
                %distance = min(sqrt(sum((predict - [CENTROID(camTarget, 1); 
                %                                    CENTROID(camTarget, 2); 
                %                                    CENTROID(camTarget, 3)]).^2, 1)));


                % Wrong prediction
                distanceWrong = ones(NB_CAMERA, NB_PERSON, N) * inf;
                for badCam = 1:NB_CAMERA
                    if badCam == camTarget || badCam == camSource
                        continue;
                    end
                    for badPerson = 1:NB_PERSON
                        if isempty(AREA{badCam}(badPerson,:)) || badPerson == pers
                            continue;
                        end

                        wrongPredict(1,:) = polyval(     AREA_TRANSFORMATION{camSource, badCam},      AREA{camSource}(pers,:), AREA_S{camSource, badCam}, AREA_U{camSource, badCam});
                        wrongPredict(2,:) = polyval(PERIMETER_TRANSFORMATION{camSource, badCam}, PERIMETER{camSource}(pers,:));
                        wrongPredict(3,:) = polyval(     MEAN_TRANSFORMATION{camSource, badCam},      MEAN{camSource}(pers,:));

                        distanceWrong(badCam, badPerson, :) = sqrt(sum((wrongPredict - [      AREA{badCam}(badPerson, 1); 
                                                                                         PERIMETER{badCam}(badPerson, 1); 
                                                                                              MEAN{badCam}(badPerson, 1)]).^2, 1));
                        %distanceWrong(badCam, badPerson, :) = sqrt(sum((wrongPredict - [CENTROID(badCam, 1); 
                        %                                                                CENTROID(badCam, 2); 
                        %                                                                CENTROID(badCam, 3)]).^2, 1));
                    end
                end
                minValue = min(distanceWrong(:));
                [idxCam, idxPerson, idxFrame] = find(distanceWrong == minValue);
                class = (minValue < distance) * idxCam(1) + (minValue >= distance) * camTarget;
                accuracy(camSource, camTarget) = accuracy(camSource, camTarget) + (class == camTarget);
            end
            accuracy(camSource, camTarget) = accuracy(camSource, camTarget) / cnt;
        end
    end

    figure("Name", 'Accuracy');
    plotConfusionMatrix(accuracy, "YOLO");
end

