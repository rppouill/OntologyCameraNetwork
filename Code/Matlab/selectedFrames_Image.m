clear all; close all; 
addpath('./tools/');

DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/Computation_Autonomous_Low_Consumption/ViewerBLE/data/SortAcquisition/Go/';
IMG_SIZE = [32, 32]; 

folderCamera = dir(fullfile(DATASET_PATH, "ESP*"));
folderCamera = folderCamera([folderCamera.isdir]);
buff_cameras = cell(length(folderCamera), 1);

for i = 1:length(folderCamera)
    folderPerson = fullfile(DATASET_PATH, folderCamera(i).name);
    subFolderPerson = dir(folderPerson); 
    subFolderPerson = subFolderPerson([subFolderPerson.isdir]); 

    buff_cameras{i} = struct();

    for j = 1:length(subFolderPerson)
        if ~strcmp(subFolderPerson(j).name, '.') && ~strcmp(subFolderPerson(j).name, '..')
            personFolder = subFolderPerson(j).name;
            buff_cameras{i}.(personFolder) = {};  
            
            subFolderPersonPath = fullfile(folderPerson, personFolder);
            pngFiles = dir(fullfile(subFolderPersonPath, '*.png'));
            
            if ~isempty(pngFiles)
                for k = 1:length(pngFiles)
                    img = imread(fullfile(subFolderPersonPath, pngFiles(k).name));
                    buff_cameras{i}.(personFolder){end+1} = img; 
                end
            end
        end
    end
end

NB_CAMERA = length(buff_cameras);
NB_PERSON = length(fieldnames(buff_cameras{1}));
Camera = cell(NB_CAMERA, NB_PERSON);
for i = 1:NB_CAMERA
    peopleFields = fieldnames(buff_cameras{i});
    
    for j = 1:NB_PERSON
        personName = peopleFields{j}; 
        images = buff_cameras{i}.(personName); 
        for k = 1:size(images,2)
            Camera{i,j}(:,:,k)= cell2mat(images(:,k));  
        end
    end
end

%% Create the Background Model
background_model = cell(NB_CAMERA,NB_PERSON);
interest    = [20,20,20];
ref         = [1,1,2];

for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        background_model{i,j} = zeros(IMG_SIZE);
        if ~isempty(Camera{i,j})
            for k = 1:interest(i)
                frame = double(im2gray(Camera{i,j}(:,:,ref(i))));
                background_model{i,j} = background_model{i,j} + frame;
            end
            background_model{i,j} = mat2gray(background_model{i,j} / interest(i));
        end
    end
end
background_model_new = background_model;
save('background_model_new.mat', 'background_model');
figure("Name", "Background Model");
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        if isempty(Camera{i,j})
            continue;
        end
        subplot(NB_CAMERA, NB_PERSON, (i-1)*NB_PERSON + j);
        imshow(background_model{i,j});
        title(['Camera ', int2str(i), ' - Person ', int2str(j)]);
    end
end

%% Compute the D vector
%% Create the foreground model and save the images into the folder
THRESH_RHO    = 0.99;       OMEGA = 2;
ALPHA         = 0.3 ; HALF_OMEGA = round(OMEGA / 2);
THRESH_MIN_PIXEL = [140, 260, 300]; 
THRESH_MAX_PIXEL = [500, 700, 500];
THRESH_SPECIAL_ANDREA = [100,200]; % Camera 3 Person 2

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
        if ~isempty(Camera{cam,pers})
            for nFrame = 1:size(Camera{cam,pers},3)
                frame = double(im2gray(Camera{cam,pers}(:,:,nFrame)));
                frame = mat2gray(frame);
                mask = zeros(size(frame));
                for h = 1:IMG_SIZE(1) - OMEGA + 1
                    for w = 1:IMG_SIZE(1) - OMEGA + 1
                        bg_patch    = background_model{cam,pers}(h:h+OMEGA-1, w:w+OMEGA-1);
                        frame_patch = frame                     (h:h+OMEGA-1, w:w+OMEGA-1);
                
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
                pixel = sum(mask(:));
                if(cam == 3 && pers == 2)
                    if pixel > THRESH_SPECIAL_ANDREA(1) && pixel < THRESH_SPECIAL_ANDREA(2)
                        N_SELECTED_FRAMES(cam,pers) = N_SELECTED_FRAMES(cam,pers) + 1;
                        SELECTED_FRAMES{cam,pers}(:,:,N_SELECTED_FRAMES(cam,pers)) = frame;
                        SELECTED_MASK  {cam,pers}(:,:,N_SELECTED_FRAMES(cam,pers)) = mask;
                        
                        MEAN_D(cam,pers) = MEAN_D(cam,pers) + pixel;
                        D{cam,pers}(end+1) = pixel;
                    end
                else
                    if pixel > THRESH_MIN_PIXEL(cam) && pixel < THRESH_MAX_PIXEL(cam)
                        N_SELECTED_FRAMES(cam,pers) = N_SELECTED_FRAMES(cam,pers) + 1;
                        SELECTED_FRAMES{cam,pers}(:,:,N_SELECTED_FRAMES(cam,pers)) = frame;
                        SELECTED_MASK  {cam,pers}(:,:,N_SELECTED_FRAMES(cam,pers)) = mask;
                        
                        MEAN_D(cam,pers) = MEAN_D(cam,pers) + pixel;
                        D{cam,pers}(end+1) = pixel;
                    end
                end
                background_model{cam,pers} = ((1-ALPHA) * background_model{cam,pers}) + (ALPHA * frame);
                MASK            {cam,pers}(:,:,nFrame)  = mask;
                FRAME_CAMERA    {cam,pers}(:,:,nFrame)  = frame;
                ALL_D           {cam,pers}(end+1)       = pixel;
            end
        end
    end
end

%% Save Selected Frames and Mask
FOLDER_PATH  = ['./Selected_Frames_Image/'];
FOLDER_FRAME = [FOLDER_PATH, 'Frames/'];
FOLDER_MASK  = [FOLDER_PATH, 'Mask/'];

if ~exist(FOLDER_FRAME, 'dir')
    mkdir(FOLDER_FRAME);
end

if ~exist(FOLDER_MASK, 'dir')
    mkdir(FOLDER_MASK);
end

for cam = 1:NB_CAMERA
    FOLDER_CAMERA_FRAME  = [FOLDER_FRAME, 'camera', int2str(cam), '/'];
    FOLDER_CAMERA_MASK   = [FOLDER_MASK , 'camera', int2str(cam), '/'];
    if ~exist(FOLDER_CAMERA_FRAME, 'dir')
        mkdir(FOLDER_CAMERA_FRAME);
    end
    if ~exist(FOLDER_CAMERA_MASK, 'dir')
        mkdir(FOLDER_CAMERA_MASK);
    end

    for pers = 1:NB_PERSON
        FOLDER_PERSON_FRAME = [FOLDER_CAMERA_FRAME , 'p', int2str(pers), '/'];
        FOLDER_PERSON_MASK  = [FOLDER_CAMERA_MASK  , 'p', int2str(pers), '/'];
        if ~exist(FOLDER_PERSON_FRAME, 'dir')
            mkdir(FOLDER_PERSON_FRAME);
        end
        if ~exist(FOLDER_PERSON_MASK, 'dir')
            mkdir(FOLDER_PERSON_MASK);
        end
        if isempty(Camera{cam,pers})
            continue;
        end
        for i = 1:N_SELECTED_FRAMES(cam,pers)
            imwrite(SELECTED_FRAMES{cam,pers}(:,:,i), [FOLDER_PERSON_FRAME, int2str(pers), '-p' sprintf('%02d', i), '.png']);
            imwrite(SELECTED_MASK  {cam,pers}(:,:,i), [FOLDER_PERSON_MASK , int2str(pers), '-p' sprintf('%02d', i), '.png']);
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
        subplot(2,3,pers);
        plot(ALL_D{cam,pers});
        if(cam == 3 && pers == 2)
            hold on; plot(ones(1,length(ALL_D{cam,pers})) * THRESH_SPECIAL_ANDREA(1), 'r--'); hold off;
            hold on; plot(ones(1,length(ALL_D{cam,pers})) * THRESH_SPECIAL_ANDREA(2), 'r--'); hold off;
        else
            hold on; plot(ones(1,length(ALL_D{cam,pers})) * THRESH_MIN_PIXEL(cam), 'r--'); hold off;
            hold on; plot(ones(1,length(ALL_D{cam,pers})) * THRESH_MAX_PIXEL(cam), 'r--'); hold off;
        end
        xlim([1, length(ALL_D{cam,pers})]);
        ylim([0, prod(IMG_SIZE)]);
        xlabel('Frame');
        ylabel('Number of Pixels');
        title(['Person ', int2str(pers)]);
        grid on;
        grid minor;
    end
end


SHOWING = false;
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