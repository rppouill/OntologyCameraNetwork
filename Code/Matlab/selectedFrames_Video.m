clear all; close all; 
addpath('./tools/');

DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/DataSet/Lobna_Dataset/9393062-Rearange/';
NB_CAMERA = 4; NB_PERSON = 9;
IMG_SIZE = [30, 30]; % We consider the square image


Camera = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        pathfile = [DATASET_PATH, int2str(i), '-person', int2str(j), '.avi'];
        if exist(pathfile, 'file')
            Camera{i,j} = VideoReader(pathfile);
        else 
            Camera{i,j} = [];
        end
    end
end

%% Create the Background Model
background_model = cell(NB_CAMERA,NB_PERSON);
interest    = [6,17,6,13];
rep         = [1,1,1,1];

for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        if ~isempty(Camera{i,j})
            background_model{i,j} = zeros(Camera{i,j}.Height, Camera{i,j}.Width);
            for k = 1:interest(i)
                frame = double(im2gray(readFrame(Camera{i,j})));
                background_model{i,j} = background_model{i,j} + frame;
            end
            background_model{i,j} = mat2gray(background_model{i,j} / interest(i));
            Camera{i,j}.CurrentTime = 0.0;
        end
    end
end


%% Compute the D vector
THRESH_RHO    = 0.99;      OMEGA = 3;
ALPHA         = 0.1 ; HALF_OMEGA = round(OMEGA / 2);
THRESH_MIN_PIXEL = [350, 400, 150, 300];
THRESH_MAX_PIXEL = [600, 800, 600, 600];

ALL_D           = cell(NB_CAMERA, NB_PERSON);
D               = cell(NB_CAMERA, NB_PERSON);
MEAN_D          = zeros(NB_CAMERA, NB_PERSON); 
FRAME_CAMERA    = cell(NB_CAMERA,NB_PERSON);

         MASK   = cell(NB_CAMERA,NB_PERSON);
SELECTED_MASK   = cell(NB_CAMERA,NB_PERSON);

N_SELECTED_FRAMES = zeros(NB_CAMERA, NB_PERSON);
  SELECTED_FRAMES =  cell(NB_CAMERA, NB_PERSON);     
SE = strel('disk', 2, 0);
for cam = 1:NB_CAMERA
    for pers = 1:NB_PERSON
        if isempty(Camera{cam,pers})
            continue;
        end
        D{cam,pers} = [];
        MASK{cam,pers}          = zeros(Camera{cam,pers}.Height, Camera{cam,pers}.Width, Camera{cam,pers}.NumberOfFrames);
        FRAME_CAMERA{cam,pers}  = zeros(Camera{cam,pers}.Height, Camera{cam,pers}.Width, Camera{cam,pers}.NumberOfFrames);
        nFrame = 1;
        while hasFrame(Camera{cam,pers})
            frame = double(im2gray(readFrame(Camera{cam,pers})));
            frame = mat2gray(frame);
            mask  = zeros(size(frame));

            for h = 1:IMG_SIZE(1) - OMEGA + 1
                for w = 1:IMG_SIZE(2) - OMEGA + 1
                    bg_patch = background_model{cam,pers}(h:h+OMEGA-1, w:w+OMEGA-1);
                    frame_patch = frame(h:h+OMEGA-1, w:w+OMEGA-1);

                    sum1 = sum(   bg_patch(:) .* frame_patch(:));
                    sum2 = sum(   bg_patch(:) .^ 2);
                    sum3 = sum(frame_patch(:) .^ 2);

                    rho = (sum1^2) / (sum2 * sum3);

                    if rho < THRESH_RHO
                        mask(h+HALF_OMEGA, w+HALF_OMEGA) = 1;
                    end
                end
            end
            mask = bwareaopen(mask, 4);
            mask = imclose(mask, strel('line', 3, 90));
            mask = imclose(mask, strel('disk', 2, 0));
            mask = imopen(mask, strel('line', 10, 90));
             % Get biggest particules
             stats = regionprops(mask, 'Area', 'BoundingBox');
             if ~isempty(stats)
                [~, idxMax] = max([stats.Area]);
                stats = stats(idxMax);
                buffMask = mask;
                mask = zeros(size(mask));
                mask(round(stats.BoundingBox(2)):floor(stats.BoundingBox(2)+stats.BoundingBox(4)), ...
                     round(stats.BoundingBox(1)):floor(stats.BoundingBox(1)+stats.BoundingBox(3))) = ...
                     buffMask(round(stats.BoundingBox(2)):floor(stats.BoundingBox(2)+stats.BoundingBox(4)), ...
                     round(stats.BoundingBox(1)):floor(stats.BoundingBox(1)+stats.BoundingBox(3)));
                 mask = bwconvhull(mask);
             end
            pixel   = sum(mask(:));
            if pixel > THRESH_MIN_PIXEL(cam) && pixel < THRESH_MAX_PIXEL(cam)
                N_SELECTED_FRAMES(cam,pers) = N_SELECTED_FRAMES(cam,pers) + 1;
                SELECTED_FRAMES{cam,pers}(:,:,N_SELECTED_FRAMES(cam,pers)) = frame;
                SELECTED_MASK  {cam,pers}(:,:,N_SELECTED_FRAMES(cam,pers)) = mask;
                MEAN_D(cam,pers) = MEAN_D(cam,pers) + pixel;
                D               {cam,pers}(end+1)       = pixel;
            end
            MASK            {cam,pers}(:,:,nFrame)  = mask;
            FRAME_CAMERA    {cam,pers}(:,:,nFrame)  = frame;
            ALL_D           {cam,pers}(end+1)       = pixel;
            
            %background_model{cam,pers}              = (1 - ALPHA) * background_model{cam,pers} + ALPHA * frame;
            nFrame                                  = nFrame + 1;
        end
        MEAN_D(cam,pers) = MEAN_D(cam,pers) / N_SELECTED_FRAMES(cam,pers);
    end
end

%% Save Selected Frames and Mask
FOLDER_PATH  = ['./Selected_Frames/'];
FOLDER_FRAME = [FOLDER_PATH, 'Frames/'];
FOLDER_MASK  = [FOLDER_PATH, 'Mask/'];

if ~exist(FOLDER_FRAME, 'dir')
    mkdir(FOLDER_FRAME);
end

if ~exist(FOLDER_MASK, 'dir')
    mkdir(FOLDER_MASK);
end

for cam = 1:NB_CAMERA
    FOLDER_CAMERA = [FOLDER_FRAME, 'camera', int2str(cam), '/'];
    FOLDER_MASK   = [FOLDER_MASK , 'camera', int2str(cam), '/'];
    if ~exist(FOLDER_CAMERA, 'dir')
        mkdir(FOLDER_CAMERA);
    end
    if ~exist(FOLDER_MASK, 'dir')
        mkdir(FOLDER_MASK);
    end

    for pers = 1:NB_PERSON
        FOLDER_PERSON = [FOLDER_CAMERA, 'p', int2str(pers), '/'];
        FOLDER_MASK   = [FOLDER_MASK  , 'p', int2str(pers), '/'];
        if ~exist(FOLDER_PERSON, 'dir')
            mkdir(FOLDER_PERSON);
        end
        if ~exist(FOLDER_MASK, 'dir')
            mkdir(FOLDER_MASK);
        end
        if isempty(Camera{cam,pers})
            continue;
        end
        for i = 1:N_SELECTED_FRAMES(cam,pers)
            imwrite(SELECTED_FRAMES{cam,pers}(:,:,i), [FOLDER_PERSON, int2str(pers), '-p' sprintf('%02d', i), '.png']);
            imwrite(SELECTED_MASK  {cam,pers}(:,:,i), [FOLDER_MASK  , int2str(pers), '-p' sprintf('%02d', i), '.png']);
        end
    end
end

save([FOLDER_PATH, 'Selected_Frames.mat'],  'SELECTED_FRAMES', ...
                                            'N_SELECTED_FRAMES', ...
                                            'SELECTED_MASK', ...
                                            'FRAME_CAMERA', ...
                                            'Camera', ...
                                            'MASK', ...
                                            'D', ...
                                            'MEAN_D', ...
                                            'background_model', ...
                                            'NB_CAMERA', 'NB_PERSON', 'IMG_SIZE');


%Display D
for cam = 1:NB_CAMERA
    figure("Name", ['D - Camera ', int2str(cam)]);
    for pers = 1:NB_PERSON
        if isempty(Camera{cam,pers})
            continue;
        end
        subplot(3,3,pers);
        plot(ALL_D{cam,pers});
        hold on; plot(ones(1,length(ALL_D{cam,pers})) * THRESH_MIN_PIXEL(cam), 'r--'); hold off;
        hold on; plot(ones(1,length(ALL_D{cam,pers})) * THRESH_MAX_PIXEL(cam), 'r--'); hold off;
        xlim([1, length(ALL_D{cam,pers})]);
        ylim([0, prod(IMG_SIZE)]);
        xlabel('Frame');
        ylabel('Number of Pixels');
        title(['Person ', int2str(pers)]);
        grid on;
        grid minor;
    end
end



SHOWING = true;
if SHOWING
    % Display 5 Selected Frames
    for cam = 1:NB_CAMERA
        figure("Name", ['Selected Frames - Camera ', int2str(cam)]);
        for pers = 1:NB_PERSON
            if isempty(SELECTED_FRAMES{cam,pers})
                continue;
            end
            for i = 1:5
                if i > N_SELECTED_FRAMES(cam,pers)
                    continue;
                end
                subplot(NB_PERSON, 5, ((pers-1)*5) + i); imshow(SELECTED_FRAMES{cam,pers}(:,:,i));
            end
        end
    end

    % Display 5 Selected Mask
    for cam = 1:NB_CAMERA
        figure("Name", ['Selected Mask - Camera ', int2str(cam)]);
        for pers = 1:NB_PERSON
            if isempty(SELECTED_MASK{cam,pers})
                continue;
            end
            for i = 1:5
                if i > N_SELECTED_FRAMES(cam,pers)
                    continue;
                end
                subplot(NB_PERSON, 5, (pers-1)*5 + i); imshow(SELECTED_MASK{cam,pers}(:,:,i));
            end
        end
    end
end
